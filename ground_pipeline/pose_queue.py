"""姿势推理 GPU 任务队列（单 worker 串行、有界等待、可取消、可观测）。

设计要点（详见 ``计划_姿势推理服务队列_20261026.md``）：

- GPU 只能串行：固定 1 个 worker 线程按「优先级 + 入队序号」出队执行。
- 有界：``capacity`` 只统计 ``pending``，满载新请求立即拒绝（不再让 HTTP 线程赌锁）。
- 公平：同一 ``client_key`` 的 pending+running 不超过 ``per_client_limit``；
  低优先级任务等待超过 ``aging_s`` 后逐步提升有效优先级，避免饿死。
- 可取消：``pending`` 可立即撤销，``running`` 只置 ``cancel`` 事件由管线检查点放弃。
- 本模块不依赖 FastAPI / torch，可用假 executor 做纯单测。
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

STATE_PENDING = "pending"
STATE_RUNNING = "running"
STATE_DONE = "done"
STATE_CANCELLED = "cancelled"
STATE_TIMEOUT = "timeout"

# 服务端内部优先级（越小越优先）；客户端不可指定
PRIORITY_INTERACTIVE = 0
PRIORITY_GROUND = 5
PRIORITY_MAX = 9

SAMPLE_WINDOW = 100  # avg/p95 统计窗口

WAIT_BUCKETS: Tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0)
EXEC_BUCKETS: Tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0)


class QueueRejected(Exception):
    """入队被拒（容量或客户端配额），由调用方映射为 503。"""

    def __init__(self, reason: str, snapshot: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.snapshot: Dict[str, Any] = dict(snapshot or {})


class QueueTimeout(Exception):
    """排队等待超时（不约束执行），由调用方映射为 503。"""

    def __init__(self, waited_ms: float, snapshot: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(f"queue wait timeout after {waited_ms:.0f} ms")
        self.waited_ms = float(waited_ms)
        self.snapshot: Dict[str, Any] = dict(snapshot or {})


@dataclass
class PoseJob:
    """一次排队中的推理任务。"""

    request_id: str
    client_key: str
    priority: int
    seq: int
    enqueued_at: float
    ahead: int = 0
    eta_ms: float = 0.0
    state: str = STATE_PENDING
    cancel: threading.Event = field(default_factory=threading.Event)
    started: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    result: Optional[Tuple[int, Dict[str, Any]]] = None
    payload: Any = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    pending_at_start: int = 0


def _fmt_bound(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _escape_label(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class PoseQueue:
    """单 worker 的 FIFO/优先级队列；线程安全。"""

    def __init__(
        self,
        *,
        capacity: int = 16,
        per_client_limit: int = 4,
        wait_timeout_s: float = 60.0,
        aging_s: float = 30.0,
        exec_timeout_s: float = 300.0,
    ) -> None:
        self.capacity = max(0, int(capacity))
        self.per_client_limit = max(0, int(per_client_limit))
        self.wait_timeout_s = max(0.0, float(wait_timeout_s))
        self.aging_s = max(0.0, float(aging_s))
        self.exec_timeout_s = max(0.0, float(exec_timeout_s))

        self._cond = threading.Condition()
        self._pending: List[PoseJob] = []
        self._running: Optional[PoseJob] = None
        self._seq = 0
        self._executor: Optional[Callable[[PoseJob], Tuple[int, Dict[str, Any]]]] = None
        self._worker: Optional[threading.Thread] = None
        self._stop = False

        self._stats: Dict[str, int] = {
            "admitted": 0,
            "completed": 0,
            "rejected_full": 0,
            "rejected_quota": 0,
            "timed_out": 0,
            "cancelled_pending": 0,
            "cancelled_running": 0,
        }
        self._completed_by_status: Dict[str, int] = {"success": 0, "error": 0, "cancelled": 0}
        self._wait_samples: Deque[float] = deque(maxlen=SAMPLE_WINDOW)
        self._exec_samples: Deque[float] = deque(maxlen=SAMPLE_WINDOW)
        self._wait_counts: List[int] = [0] * len(WAIT_BUCKETS)
        self._exec_counts: List[int] = [0] * len(EXEC_BUCKETS)
        self._wait_sum = 0.0
        self._wait_count = 0
        self._exec_sum = 0.0
        self._exec_count = 0

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        """``capacity=0`` 表示禁用排队（回滚用）：忙时直接拒绝。"""
        return self.capacity > 0

    @property
    def worker_alive(self) -> bool:
        worker = self._worker
        return worker is not None and worker.is_alive()

    def start(self, executor: Callable[[PoseJob], Tuple[int, Dict[str, Any]]]) -> None:
        """启动 worker；``executor(job) -> (http_code, body)`` 在 worker 线程内串行调用。"""
        with self._cond:
            if self._worker is not None and self._worker.is_alive():
                return
            self._executor = executor
            self._stop = False
            self._worker = threading.Thread(target=self._run, name="pose-queue-worker", daemon=True)
            self._worker.start()
        logger.info(
            "pose_queue start worker capacity=%s per_client=%s wait_timeout=%ss aging=%ss",
            self.capacity,
            self.per_client_limit,
            self.wait_timeout_s,
            self.aging_s,
        )

    def shutdown(self, join_timeout: float = 5.0) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()
            worker = self._worker
        if worker is not None:
            worker.join(timeout=join_timeout)
        logger.info("pose_queue stop worker")

    # ------------------------------------------------------------------
    # 入队 / 等待 / 取消
    # ------------------------------------------------------------------
    def submit(
        self,
        *,
        request_id: str,
        client_key: str,
        priority: int = PRIORITY_INTERACTIVE,
        payload: Any = None,
    ) -> PoseJob:
        rid = str(request_id or "").strip()
        client = str(client_key or "").strip() or "unknown"
        prio = min(max(int(priority), 0), PRIORITY_MAX)
        with self._cond:
            if not self.enabled:
                if self._running is not None or self._pending:
                    self._stats["rejected_full"] += 1
                    logger.warning(
                        "pose_queue reject rid=%s reason=gpu_busy pending=%s running=%s",
                        rid,
                        len(self._pending),
                        self._running.request_id if self._running is not None else "",
                    )
                    raise QueueRejected("gpu_busy", self._snapshot_locked())
            elif len(self._pending) >= self.capacity:
                self._stats["rejected_full"] += 1
                logger.warning(
                    "pose_queue reject rid=%s reason=queue_full pending=%s capacity=%s",
                    rid,
                    len(self._pending),
                    self.capacity,
                )
                raise QueueRejected("queue_full", self._snapshot_locked())
            if self.per_client_limit > 0 and self._client_count_locked(client) >= self.per_client_limit:
                self._stats["rejected_quota"] += 1
                logger.warning(
                    "pose_queue reject rid=%s reason=client_quota client=%s limit=%s",
                    rid,
                    client,
                    self.per_client_limit,
                )
                raise QueueRejected("client_quota", self._snapshot_locked())

            self._seq += 1
            job = PoseJob(
                request_id=rid,
                client_key=client,
                priority=prio,
                seq=self._seq,
                enqueued_at=time.monotonic(),
                payload=payload,
            )
            job.ahead = len(self._pending) + (1 if self._running is not None else 0)
            job.eta_ms = round(job.ahead * self._avg_exec_ms_locked(), 1)
            self._pending.append(job)
            self._stats["admitted"] += 1
            self._cond.notify_all()
        logger.info(
            "pose_queue enqueue rid=%s client=%s priority=%s ahead=%s pending=%s",
            rid,
            client,
            prio,
            job.ahead,
            len(self._pending),
        )
        return job

    def wait(
        self, job: PoseJob, timeout_s: Optional[float] = None
    ) -> Optional[Tuple[int, Dict[str, Any]]]:
        """阻塞等待结果。

        - 排队超过 ``timeout_s``（与服务默认取小）→ 撤销排队并抛 :class:`QueueTimeout`；
        - pending 阶段被取消 → 返回 ``None``（由调用方拼 status=cancelled 响应）；
        - 已开始执行 → 不设上限，等 worker 完成。
        """
        limit = self.wait_timeout_s if timeout_s is None else min(float(timeout_s), self.wait_timeout_s)
        deadline = time.monotonic() + max(0.0, limit)
        while not job.done.is_set():
            if job.state == STATE_RUNNING:
                job.done.wait()
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if self._withdraw(job):
                    waited_ms = (time.monotonic() - job.enqueued_at) * 1000.0
                    with self._cond:
                        self._stats["timed_out"] += 1
                    logger.info("pose_queue timeout rid=%s waited_ms=%.1f", job.request_id, waited_ms)
                    raise QueueTimeout(waited_ms, self.snapshot())
                continue
            job.done.wait(min(0.05, remaining))
        if job.state == STATE_CANCELLED and job.result is None:
            return None
        return job.result

    def cancel(self, request_id: str) -> Dict[str, Any]:
        """按 ``request_id`` 取消；返回 ``{found, state, ahead}``。"""
        rid = str(request_id or "").strip()
        if not rid:
            return {"found": False, "state": "", "ahead": 0}
        with self._cond:
            for job in self._pending:
                if job.request_id != rid:
                    continue
                self._pending.remove(job)
                job.state = STATE_CANCELLED
                job.finished_at = time.monotonic()
                job.cancel.set()
                job.done.set()
                self._stats["cancelled_pending"] += 1
                self._cond.notify_all()
                logger.info("pose_queue cancel rid=%s state=pending ahead=%s", rid, job.ahead)
                return {"found": True, "state": STATE_PENDING, "ahead": job.ahead}
            running = self._running
            if running is not None and running.request_id == rid:
                running.cancel.set()
                self._stats["cancelled_running"] += 1
                logger.info("pose_queue cancel rid=%s state=running", rid)
                return {"found": True, "state": STATE_RUNNING, "ahead": 0}
        return {"found": False, "state": "", "ahead": 0}

    # ------------------------------------------------------------------
    # 观测
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        with self._cond:
            return self._snapshot_locked()

    def summary(self) -> Dict[str, Any]:
        """health 用的紧凑摘要。"""
        with self._cond:
            return {
                "enabled": self.enabled,
                "capacity": self.capacity,
                "pending": len(self._pending),
                "running": self._running is not None,
            }

    def render_prometheus(self, *, client_label: bool = True) -> str:
        """手写 Prometheus 文本格式（避免引入 ``prometheus_client`` 依赖）。"""
        with self._cond:
            snapshot = self._snapshot_locked()
            running = 1 if self._running is not None else 0
            per_client = dict(snapshot["per_client"])
            stats = dict(snapshot["stats"])
            completed = dict(self._completed_by_status)
            wait_counts = list(self._wait_counts)
            exec_counts = list(self._exec_counts)
            wait_sum, wait_count = self._wait_sum, self._wait_count
            exec_sum, exec_count = self._exec_sum, self._exec_count

        lines: List[str] = []

        def gauge(name: str, help_text: str, value: Any, labels: str = "") -> None:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name}{labels} {value}")

        def counter(name: str, help_text: str) -> None:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} counter")

        gauge("pose_queue_capacity", "队列等待位容量（0 表示禁用排队）", self.capacity)
        gauge("pose_queue_pending", "当前等待中的任务数", snapshot["pending"])
        gauge("pose_queue_running", "GPU 是否有任务在执行（0/1）", running)
        if client_label:
            lines.append("# HELP pose_queue_per_client 各客户端占用的 pending+running 任务数")
            lines.append("# TYPE pose_queue_per_client gauge")
            for client, count in sorted(per_client.items()):
                lines.append(f'pose_queue_per_client{{client="{_escape_label(client)}"}} {count}')

        counter("pose_queue_admitted_total", "累计入队任务数")
        lines.append(f"pose_queue_admitted_total {stats['admitted']}")
        counter("pose_queue_completed_total", "累计完成任务数（按结果分类）")
        for status in ("success", "error", "cancelled"):
            lines.append(f'pose_queue_completed_total{{status="{status}"}} {completed.get(status, 0)}')
        counter("pose_queue_rejected_total", "累计拒绝数（按原因分类）")
        lines.append(f'pose_queue_rejected_total{{reason="queue_full"}} {stats["rejected_full"]}')
        lines.append(f'pose_queue_rejected_total{{reason="client_quota"}} {stats["rejected_quota"]}')
        counter("pose_queue_timed_out_total", "累计排队超时数")
        lines.append(f"pose_queue_timed_out_total {stats['timed_out']}")
        counter("pose_queue_cancelled_total", "累计取消数（按任务状态分类）")
        lines.append(f'pose_queue_cancelled_total{{state="pending"}} {stats["cancelled_pending"]}')
        lines.append(f'pose_queue_cancelled_total{{state="running"}} {stats["cancelled_running"]}')

        lines.extend(
            self._render_histogram(
                "pose_queue_wait_seconds",
                "排队等待时长（秒）",
                WAIT_BUCKETS,
                wait_counts,
                wait_sum,
                wait_count,
            )
        )
        lines.extend(
            self._render_histogram(
                "pose_queue_exec_seconds",
                "任务执行时长（秒）",
                EXEC_BUCKETS,
                exec_counts,
                exec_sum,
                exec_count,
            )
        )
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    @staticmethod
    def _render_histogram(
        name: str,
        help_text: str,
        buckets: Tuple[float, ...],
        counts: List[int],
        total_sum: float,
        total_count: int,
    ) -> List[str]:
        lines = [f"# HELP {name} {help_text}", f"# TYPE {name} histogram"]
        cumulative = 0
        for bound, count in zip(buckets, counts):
            cumulative += count
            lines.append(f'{name}_bucket{{le="{_fmt_bound(bound)}"}} {cumulative}')
        lines.append(f'{name}_bucket{{le="+Inf"}} {total_count}')
        lines.append(f"{name}_sum {round(total_sum, 6)}")
        lines.append(f"{name}_count {total_count}")
        return lines

    def _client_count_locked(self, client_key: str) -> int:
        count = sum(1 for job in self._pending if job.client_key == client_key)
        if self._running is not None and self._running.client_key == client_key:
            count += 1
        return count

    def _snapshot_locked(self) -> Dict[str, Any]:
        running = None
        if self._running is not None:
            exec_ms = 0.0
            if self._running.started_at is not None:
                exec_ms = (time.monotonic() - self._running.started_at) * 1000.0
            running = {
                "request_id": self._running.request_id,
                "client": self._running.client_key,
                "exec_ms": round(exec_ms, 1),
                "overdue": bool(self.exec_timeout_s > 0 and exec_ms / 1000.0 > self.exec_timeout_s),
            }
        per_client: Dict[str, int] = {}
        jobs = list(self._pending) + ([self._running] if self._running is not None else [])
        for job in jobs:
            per_client[job.client_key] = per_client.get(job.client_key, 0) + 1
        samples = sorted(self._wait_samples)
        p95 = 0.0
        if samples:
            p95 = samples[min(len(samples) - 1, int(round(0.95 * (len(samples) - 1))))]
        stats = dict(self._stats)
        stats.update(
            {
                "avg_wait_ms": round(self._avg(self._wait_samples) * 1000.0, 1),
                "p95_wait_ms": round(p95 * 1000.0, 1),
                "avg_exec_ms": round(self._avg(self._exec_samples) * 1000.0, 1),
            }
        )
        return {
            "enabled": self.enabled,
            "capacity": self.capacity,
            "pending": len(self._pending),
            "running": running,
            "per_client": per_client,
            "stats": stats,
        }

    @staticmethod
    def _avg(samples: Deque[float]) -> float:
        return float(sum(samples)) / len(samples) if samples else 0.0

    def _avg_exec_ms_locked(self) -> float:
        return self._avg(self._exec_samples) * 1000.0

    def _effective_priority_locked(self, job: PoseJob, now: float) -> int:
        if self.aging_s <= 0:
            return job.priority
        steps = int(max(0.0, now - job.enqueued_at) // self.aging_s)
        return max(0, job.priority - steps)

    def _pick_locked(self) -> Optional[PoseJob]:
        now = time.monotonic()
        best: Optional[PoseJob] = None
        best_key: Optional[Tuple[int, int]] = None
        stale: List[PoseJob] = []
        for job in self._pending:
            if job.state != STATE_PENDING:
                stale.append(job)
                continue
            key = (self._effective_priority_locked(job, now), job.seq)
            if best_key is None or key < best_key:
                best, best_key = job, key
        for job in stale:
            self._pending.remove(job)
        if best is not None:
            self._pending.remove(best)
        return best

    def _withdraw(self, job: PoseJob) -> bool:
        with self._cond:
            if job.state != STATE_PENDING:
                return False
            try:
                self._pending.remove(job)
            except ValueError:
                pass
            job.state = STATE_TIMEOUT
            job.finished_at = time.monotonic()
            job.done.set()
            self._cond.notify_all()
            return True

    def _record_locked(self, wait_ms: float, exec_ms: float, result: Optional[Tuple[int, Dict[str, Any]]]) -> None:
        status = "success"
        if result is not None:
            code, body = result
            if isinstance(body, dict) and body.get("status") == "cancelled":
                status = "cancelled"
            elif code >= 400 or (isinstance(body, dict) and body.get("status") == "error"):
                status = "error"
        self._stats["completed"] += 1
        self._completed_by_status[status] = self._completed_by_status.get(status, 0) + 1

        wait_s = max(0.0, wait_ms) / 1000.0
        exec_s = max(0.0, exec_ms) / 1000.0
        self._wait_samples.append(wait_s)
        self._exec_samples.append(exec_s)
        self._wait_sum += wait_s
        self._wait_count += 1
        self._exec_sum += exec_s
        self._exec_count += 1
        for index, bound in enumerate(WAIT_BUCKETS):
            if wait_s <= bound:
                self._wait_counts[index] += 1
                break
        for index, bound in enumerate(EXEC_BUCKETS):
            if exec_s <= bound:
                self._exec_counts[index] += 1
                break

    def _run(self) -> None:
        while True:
            with self._cond:
                while True:
                    if self._stop:
                        return
                    job = self._pick_locked()
                    if job is not None:
                        break
                    self._cond.wait(0.5)
                job.state = STATE_RUNNING
                job.started_at = time.monotonic()
                job.pending_at_start = len(self._pending)
                self._running = job
            job.started.set()

            executor = self._executor
            wait_ms = (job.started_at - job.enqueued_at) * 1000.0
            logger.info(
                "pose_queue start rid=%s wait_ms=%.1f pending=%s",
                job.request_id,
                wait_ms,
                job.pending_at_start,
            )
            started = time.perf_counter()
            try:
                if executor is None:
                    raise RuntimeError("pose queue executor 未配置")
                result: Optional[Tuple[int, Dict[str, Any]]] = executor(job)
            except Exception as exc:  # noqa: BLE001
                logger.exception("pose_queue exec 失败 rid=%s", job.request_id)
                result = (
                    500,
                    {
                        "status": "error",
                        "error": f"queue_executor: {type(exc).__name__}: {exc}",
                        "request_id": job.request_id,
                    },
                )
            exec_ms = (time.perf_counter() - started) * 1000.0

            with self._cond:
                self._running = None
                job.finished_at = time.monotonic()
                job.result = result
                job.state = STATE_DONE
                self._record_locked(wait_ms, exec_ms, result)
                job.done.set()
                self._cond.notify_all()

            status = "success"
            if result is not None:
                code, body = result
                if isinstance(body, dict) and body.get("status") == "cancelled":
                    status = "cancelled"
                elif code >= 400 or (isinstance(body, dict) and body.get("status") == "error"):
                    status = "error"
            log = logger.info if status == "success" else logger.warning
            log("pose_queue finish rid=%s exec_ms=%.1f status=%s", job.request_id, exec_ms, status)
            if self.exec_timeout_s > 0 and exec_ms / 1000.0 > self.exec_timeout_s:
                logger.warning(
                    "pose_queue stuck rid=%s exec_ms=%.1f threshold_s=%s",
                    job.request_id,
                    exec_ms,
                    self.exec_timeout_s,
                )
