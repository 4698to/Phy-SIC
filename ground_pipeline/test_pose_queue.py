"""姿势推理队列单元测试（无 GPU / 无 FastAPI 依赖）。

运行：``python -m unittest ground_pipeline.test_pose_queue -v``
"""
from __future__ import annotations

import threading
import time
import unittest

from ground_pipeline.pose_queue import PoseQueue, QueueRejected, QueueTimeout


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class _Recorder:
    """记录执行顺序；可选 gate 让任务阻塞在 worker 内。"""

    def __init__(self, *, delay: float = 0.0, gated: bool = False) -> None:
        self.order: list[str] = []
        self.delay = delay
        self.gate = threading.Event()
        if not gated:
            self.gate.set()
        self.started: list[str] = []
        self._lock = threading.Lock()
        self.running = 0
        self.max_running = 0

    def __call__(self, job):
        with self._lock:
            self.order.append(job.request_id)
            self.started.append(job.request_id)
            self.running += 1
            self.max_running = max(self.max_running, self.running)
        try:
            if self.gate is not None:
                self.gate.wait(3.0)
            if self.delay:
                time.sleep(self.delay)
            return 200, {"status": "success", "request_id": job.request_id}
        finally:
            with self._lock:
                self.running -= 1


class PoseQueueTest(unittest.TestCase):
    def _queue(self, **kwargs) -> PoseQueue:
        params = {"capacity": 8, "per_client_limit": 0, "wait_timeout_s": 5.0, "aging_s": 0.0}
        params.update(kwargs)
        queue = PoseQueue(**params)
        self.addCleanup(queue.shutdown)
        return queue

    def test_capacity_rejects_overflow(self):
        queue = self._queue(capacity=2)
        recorder = _Recorder(gated=True)
        queue.start(recorder)

        first = queue.submit(request_id="r1", client_key="a", priority=0)
        self.assertTrue(first.started.wait(1.0))
        queue.submit(request_id="r2", client_key="a", priority=0)
        queue.submit(request_id="r3", client_key="a", priority=0)
        with self.assertRaises(QueueRejected) as ctx:
            queue.submit(request_id="r4", client_key="a", priority=0)
        self.assertEqual(ctx.exception.reason, "queue_full")
        self.assertEqual(ctx.exception.snapshot["pending"], 2)
        recorder.gate.set()
        self.assertEqual(queue.wait(first)[0], 200)

    def test_fifo_same_priority(self):
        queue = self._queue()
        recorder = _Recorder()
        for rid in ("r1", "r2", "r3"):
            queue.submit(request_id=rid, client_key="a", priority=0)
        queue.start(recorder)
        self.assertEqual(recorder.started[:1], [])
        self.assertTrue(_wait_until(lambda: len(recorder.order) == 3))
        self.assertEqual(recorder.order, ["r1", "r2", "r3"])

    def test_lower_priority_value_first(self):
        queue = self._queue()
        recorder = _Recorder()
        queue.submit(request_id="ground", client_key="a", priority=5)
        queue.submit(request_id="interactive", client_key="a", priority=0)
        queue.start(recorder)
        self.assertTrue(_wait_until(lambda: len(recorder.order) == 2))
        self.assertEqual(recorder.order, ["interactive", "ground"])

    def test_aging_promotes_long_waiter(self):
        queue = self._queue(aging_s=0.02)
        recorder = _Recorder()
        queue.submit(request_id="ground", client_key="a", priority=5)
        time.sleep(0.3)
        queue.submit(request_id="interactive", client_key="a", priority=0)
        queue.start(recorder)
        self.assertTrue(_wait_until(lambda: len(recorder.order) == 2))
        self.assertEqual(recorder.order, ["ground", "interactive"])

    def test_aging_disabled_keeps_priority(self):
        queue = self._queue(aging_s=0.0)
        recorder = _Recorder()
        queue.submit(request_id="ground", client_key="a", priority=5)
        time.sleep(0.05)
        queue.submit(request_id="interactive", client_key="a", priority=0)
        queue.start(recorder)
        self.assertTrue(_wait_until(lambda: len(recorder.order) == 2))
        self.assertEqual(recorder.order, ["interactive", "ground"])

    def test_per_client_quota(self):
        queue = self._queue(capacity=8, per_client_limit=2)
        queue.submit(request_id="a1", client_key="a", priority=0)
        queue.submit(request_id="a2", client_key="a", priority=0)
        with self.assertRaises(QueueRejected) as ctx:
            queue.submit(request_id="a3", client_key="a", priority=0)
        self.assertEqual(ctx.exception.reason, "client_quota")
        queue.submit(request_id="b1", client_key="b", priority=0)
        self.assertEqual(queue.snapshot()["per_client"], {"a": 2, "b": 1})

    def test_wait_timeout_withdraws_job(self):
        queue = self._queue(capacity=4, wait_timeout_s=0.1)
        job = queue.submit(request_id="slow", client_key="a", priority=0)
        with self.assertRaises(QueueTimeout) as ctx:
            queue.wait(job)
        self.assertGreater(ctx.exception.waited_ms, 50.0)

        recorder = _Recorder()
        queue.start(recorder)
        time.sleep(0.2)
        self.assertEqual(recorder.started, [])
        self.assertEqual(queue.snapshot()["stats"]["timed_out"], 1)

    def test_cancel_pending_is_immediate(self):
        queue = self._queue()
        job = queue.submit(request_id="r1", client_key="a", priority=0)
        info = queue.cancel("r1")
        self.assertEqual(info, {"found": True, "state": "pending", "ahead": 0})
        self.assertIsNone(queue.wait(job, timeout_s=1.0))

        recorder = _Recorder()
        queue.start(recorder)
        time.sleep(0.2)
        self.assertEqual(recorder.started, [])
        self.assertEqual(queue.snapshot()["stats"]["cancelled_pending"], 1)

    def test_cancel_running_only_sets_event(self):
        queue = self._queue()
        observed = threading.Event()

        def executor(job):
            observed.wait(1.0)
            return 200, {"status": "cancelled" if job.cancel.is_set() else "success"}

        queue.start(executor)
        job = queue.submit(request_id="r1", client_key="a", priority=0)
        self.assertTrue(job.started.wait(1.0))
        info = queue.cancel("r1")
        self.assertEqual(info["state"], "running")
        observed.set()
        code, body = queue.wait(job)
        self.assertEqual(code, 200)
        self.assertEqual(body["status"], "cancelled")
        self.assertEqual(queue.snapshot()["stats"]["cancelled_running"], 1)

    def test_unknown_cancel(self):
        queue = self._queue()
        info = queue.cancel("nope")
        self.assertFalse(info["found"])

    def test_stats_and_snapshot(self):
        queue = self._queue(capacity=2)
        recorder = _Recorder(delay=0.05)
        queue.start(recorder)

        first = queue.submit(request_id="r1", client_key="a", priority=0)
        self.assertEqual(queue.wait(first)[0], 200)
        second = queue.submit(request_id="r2", client_key="a", priority=0)
        self.assertEqual(queue.wait(second)[0], 200)

        snapshot = queue.snapshot()
        self.assertEqual(snapshot["pending"], 0)
        self.assertIsNone(snapshot["running"])
        self.assertEqual(snapshot["stats"]["admitted"], 2)
        self.assertEqual(snapshot["stats"]["completed"], 2)
        # 统计字段以毫秒为单位（样本内部按秒存储，输出前换算）
        self.assertGreaterEqual(snapshot["stats"]["avg_exec_ms"], 40.0)
        self.assertLess(snapshot["stats"]["avg_exec_ms"], 1000.0)
        self.assertEqual(queue.summary()["capacity"], 2)

    def test_disabled_queue_rejects_when_busy(self):
        queue = self._queue(capacity=0)
        recorder = _Recorder(gated=True)
        queue.start(recorder)
        first = queue.submit(request_id="r1", client_key="a", priority=0)
        self.assertTrue(first.started.wait(1.0))
        with self.assertRaises(QueueRejected) as ctx:
            queue.submit(request_id="r2", client_key="a", priority=0)
        self.assertEqual(ctx.exception.reason, "gpu_busy")
        recorder.gate.set()
        queue.wait(first)

    def test_prometheus_render(self):
        queue = self._queue(capacity=2, per_client_limit=0)

        def executor(job):
            if job.request_id == "bad":
                return 500, {"status": "error", "error": "boom"}
            return 200, {"status": "success"}

        queue.start(executor)
        ok = queue.submit(request_id="ok", client_key="c1", priority=0)
        queue.wait(ok)
        bad = queue.submit(request_id="bad", client_key="c1", priority=0)
        queue.wait(bad)

        text = queue.render_prometheus(client_label=True)
        self.assertIn("pose_queue_admitted_total 2", text)
        self.assertIn('pose_queue_completed_total{status="success"} 1', text)
        self.assertIn('pose_queue_completed_total{status="error"} 1', text)
        self.assertIn('pose_queue_rejected_total{reason="queue_full"} 0', text)
        self.assertIn('pose_queue_wait_seconds_bucket{le="+Inf"} 2', text)
        self.assertIn('pose_queue_exec_seconds_bucket{le="+Inf"} 2', text)
        self.assertIn('pose_queue_pending 0', text)

        no_label = queue.render_prometheus(client_label=False)
        self.assertNotIn("client=", no_label)
        self.assertIn("pose_queue_capacity 2", no_label)

    def test_prometheus_per_client_label(self):
        queue = self._queue(capacity=2, per_client_limit=0)
        job = queue.submit(request_id="r1", client_key="c1", priority=0)
        text = queue.render_prometheus(client_label=True)
        self.assertIn('pose_queue_per_client{client="c1"} 1', text)
        self.assertIn('pose_queue_pending 1', text)
        queue.cancel("r1")
        queue.wait(job, timeout_s=1.0)

    def test_concurrent_submit_is_serial(self):
        queue = self._queue(capacity=32, per_client_limit=0)
        recorder = _Recorder(delay=0.005)
        queue.start(recorder)
        results: list[int] = []
        lock = threading.Lock()

        def worker(index: int) -> None:
            job = queue.submit(request_id=f"r{index:02d}", client_key=f"c{index}", priority=0)
            code, _ = queue.wait(job)
            with lock:
                results.append(code)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10.0)

        self.assertEqual(results, [200] * 20)
        self.assertEqual(len(recorder.order), 20)
        self.assertEqual(recorder.max_running, 1)
        self.assertEqual(queue.snapshot()["stats"]["completed"], 20)


if __name__ == "__main__":
    unittest.main()
