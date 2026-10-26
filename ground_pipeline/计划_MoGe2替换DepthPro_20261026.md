# 计划：用 MoGe-2 替换 DepthPro（加速与姿势精度评估）

- 日期：2026-10-26
- 状态：**调研完成，待评审；未开工**
- 范围：`ground_pipeline` 地面/姿势推理服务中的**深度模型**替换，不含分割模型（LangSAM）、姿势模型（SAM3D Body）替换
- 结论先行见 §0；尚未实测的部分全部显式标注「待实测」，本文不给出未经验证的加速倍数

---

## 0. 结论摘要（TL;DR）

| 问题 | 结论 | 依据强度 |
| --- | --- | --- |
| MoGe-2 能否在能力上替代 DepthPro？ | **能**。MoGe-2 一次前向同时给出米制深度 + 归一化相机内参 + valid mask，接口上覆盖 DepthPro 的 `depth` + `focallength_px` 两项输出 | 官方 README / 论文摘要，已验证 |
| 能否加速？ | **有条件地可能，但不保证**。官方声明 MoGe-2 ViT-L 约 60 ms/图（A100/RTX 3090，FP16），DepthPro 是固定 1536×1536 输入（约 300 ms/图，V100 量级）。但现有代码**只有总耗时 `time_ms`，没有分段计时**，无法判断深度阶段占整条链路的百分比；必须先做 P0 计时，再决定是否值得换 | 官方声明 + 代码现状，**待本机实测** |
| 会不会掉姿势精度？ | **主路径大概率不掉，回退路径与地面拟合需要重标定**。主路径 `camera_ground_anchor` 尺度恒为 1.0，深度尺度误差不进入网格尺寸；深度只影响地平面高度/倾角 → 世界系高度与接触。真正风险是三处与深度噪声强耦合的经验常数：`extract_ground` 的 2 cm 阈值、`_keypoints_depth_camera` 的 `spread/0.04` 权重、回退 Umeyama 的尺度区间 | 代码推导，**待三图回归实测** |
| 建议动作 | **P0 先加分段计时与深度阶段基准脚本（不改模型）→ 只有当深度阶段占比 ≥ 25% 且 MoGe-2 实测显著更快时，才进入 P1 适配层实现** | 本计划 |

一句话：**这件事的瓶颈判断目前是盲的**。先量，再换；换的时候用适配层 + 开关 + 灰度，不用一把梭。

---

## 1. 背景与现状

### 1.1 服务链路

`ground_service.py` 单进程 FastAPI + 单张 GPU，模型串行：

```
RGB ──► LangSAM(groundingdino+sam2.1) ──► floor_mask
    └──► DepthPro(1536×1536 → 原图分辨率 depth + f_px) ──► depth, K
                └──► extract_ground(RANSAC+SVD) ──► plane_normal, plane_d
                            ├──► /v1/ground 响应（点云、contact_pixels、depth_b64…）
                            └──► GroundTransform ──► SAM3D Body ──► 落地/对齐/重定向
```

### 1.2 DepthPro 的全部消费点

| # | 用途 | 位置 |
| --- | --- | --- |
| 1 | 启动加载（先于 LangSAM，WSL 稳定性硬约束） | [ground_service.py](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L2334-L2343) |
| 2 | `/v1/ground` 主推理：`transform(rgb)` → `model.infer` → `depth` + `focallength_px` | [ground_service.py](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L2451-L2463) |
| 3 | `/v1/pose/*` 与 `/v2/pose/*` 的地面变换：同一套调用，再喂 `extract_ground` | [ground_service.py](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L1581-L1613) |
| 4 | 关键点处深度采样（回退路径与残差诊断的目标点） | [ground_service.py](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L350-L385) |
| 5 | 可视化与元数据：`DepthPro scene cloud` 图例、`method: weighted_keypoint_umeyama+depthpro`、注释/docstring | [ground_service.py](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L1128-L1137)、[L906](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L900-L910)、[L662](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L660-L664) |

深度数组在下游被间接使用的位置：`person_bbox_3d_m` / `person_centroid_m`（[L2505-L2516](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L2505-L2516)）、`contact_pixels`（[L2518 起](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L2518-L2530)）、`depth_range_m`（[L2724](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L2724)）、`depth_b64`（[L2557](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L2555-L2560)）。

**这些位置不需要改代码**——只要新深度模型仍然输出「原图分辨率的米制 depth + 3×3 像素内参 K」，下游全部无感。

### 1.3 现有耗时观测能力（关键缺口）

全文件只有两处计时：

- `/v1/ground`：`time_ms` = 整个请求总耗时（[L2279](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L2279)）
- `/v1|v2/pose/single`：`pose["time_ms"]` 只覆盖 SAM3D 那一段（[L2485-L2493](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L2485-L2493)）

**LangSAM 分割、深度推理、`extract_ground` 三段都没有单独计时。** 因此「DepthPro 是不是瓶颈」目前无法从线上数据回答，这是本计划的第一个前置动作。

### 1.4 仓库里已有的 MoGe 痕迹（重要）

MoGe 不是陌生依赖，本仓库全量流程已经在用它：

| 位置 | 内容 |
| --- | --- |
| [requirements.txt](file:///g:/UGit/Phy-SIC/requirements.txt#L12) | 已 pin `git+https://github.com/microsoft/MoGe.git@0286b49...` |
| [utils/depth.py](file:///g:/UGit/Phy-SIC/utils/depth.py#L18-L34) | `from moge.model.v1 import MoGeModel`；`MoGeModel.from_pretrained("Ruicheng/moge-vitl")` |
| [utils/depth.py](file:///g:/UGit/Phy-SIC/utils/depth.py#L107-L124) | `run_metric_moge_from_depthpro`：**用 DepthPro 把 MoGe 的 depth 对齐成米制** |
| [utils/depth.py](file:///g:/UGit/Phy-SIC/utils/depth.py#L156-L163) | 注释：`This is only required for MoGe v1: v2 returns metric points.` |
| [cfg/v1.yaml](file:///g:/UGit/Phy-SIC/cfg/v1.yaml#L40) | `f_px: moge` —— 全量流程用 MoGe 的内参选支 |

两条结论：

1. **MoGe-2 的「米制」特性恰好消掉了历史代码里「MoGe 只给相对深度、必须用 DepthPro 对齐」的那个补丁**——这正是替换在原理上成立的地方。
2. 现有 `moge` 依赖 pin 的是 v1 时期的 commit，`moge.model.v2` 是否存在需在目标环境核对（见 §9 待验证项）。另外轻量化服务环境（WSL conda `ground-wsl`）**当前没有装 moge**（已实测：`ModuleNotFoundError: No module named 'moge'`）。

---

## 2. 待回答的问题

| 编号 | 问题 | 回答方式 |
| --- | --- | --- |
| Q1 | 深度阶段在 `/v1/ground` 与 `/v1/pose/single` 中各占多少毫秒/百分比？ | P0 分段计时 + 单图基准脚本 |
| Q2 | 在目标 GPU（RTX 4070 12 GB）上，MoGe-2 ViT-L / ViT-B 实测多少毫秒、多少显存？ | P0 基准脚本（FP32 / FP16 各一档） |
| Q3 | MoGe-2 的米制深度的绝对尺度误差是多少？与 DepthPro 相比在同一张图上差多少？ | 三张基线图 + 已知身高/已知相机高度做对照 |
| Q4 | 换模型后地平面法向/高度漂移多少？姿势指标是否劣化？ | 三图回归，逐项对阈值 |
| Q5 | 显存叠加：DepthPro(≈2.5 GB) 换成 MoGe-2 后，与 SAM3D ViT-H 共存是否更宽裕？ | 实测 `torch.cuda.max_memory_allocated` |

---

## 3. MoGe-2 能力与约束

### 3.1 模型档位（官方 README）

| 模型 | 参数量 | 说明 |
| --- | --- | --- |
| `moge-2-vitl` | 326 M | 米制尺度，无 normal |
| `moge-2-vitl-normal` | 331 M | 米制尺度 + normal |
| `moge-2-vitb-normal` | 104 M | 米制尺度 + normal，**速度优先的候选** |
| `moge-2-vits-normal` | 35 M | 最小档，精度存疑，仅作下限参考 |

对比：DepthPro ≈ 646.5 M 参数、输入固定 1536×1536。MoGe-2 ViT-L 参数量约为 DepthPro 的一半，ViT-B 约为其 1/6。

### 3.2 输出字段

`model.infer(...)` 返回 dict（字段名以本地安装版本为准，下列为 README 语义）：

| 字段 | 语义 | 本项目用途 |
| --- | --- | --- |
| `points` | 米制点图（H×W×3，相机系） | 可绕过 `depth_to_points` 直接用，但**不必**（保持现有 `extract_ground` 入参不变更省事） |
| `depth` | 米制 z 深度（H×W） | 替换 `pred["depth"]` |
| `normal` | 表面法向（仅 `*-normal` 档） | **本期不用**，属范围外 |
| `intrinsics` | **归一化** 3×3 内参（fx/W、fy/H、cx/W、cy/H） | 替换 `pred["focallength_px"]`，需转像素 |
| `valid_mask` | 有效像素掩码（天空/无效区为 False） | 可传给 `extract_ground` 前做无效值清理（非必需，`extract_ground` 已有 `isfinite` 过滤） |

### 3.3 速度声明（必须本机复核）

- 官方 README：ViT-L 约 **60 ms/图**，前提是 A100 或 RTX 3090 + FP16。
- 可通过 `num_tokens`（建议 1200–2500）与 `resolution_level` 进一步降算力。
- **不要直接引用这个数字到验收结论**：分辨率、精度模式、GPU 型号、是否含预处理/后处理都会改变结果。

### 3.4 环境与许可

| 项 | MoGe-2 | DepthPro |
| --- | --- | --- |
| 代码许可 | MIT（`moge/model/modules/dinov2` 为 Apache 2.0） | Apple ASCL，**非商业** |
| 权重许可 | MIT | ASCL |
| Python | ≥ 3.10 | — |
| 本项目环境 | WSL conda `ground-wsl` 为 Python 3.10（[WSL_SETUP.md](file:///g:/UGit/Phy-SIC/ground_pipeline/WSL_SETUP.md#L30-L49)），**版本满足**；但 `moge` 未安装 | 已安装并 `pip install -e` |

许可变化是本次替换的**附带收益**：DepthPro 的非商业条款对商用交付是硬约束，MoGe-2 换成 MIT 后这条风险消失。这一条独立于加速与精度，值得单独写进决策依据。

### 3.5 版本边界

最新已到 MoGe-3（1.25 B 参数 ViT-G，依赖 FlexGEMM/Triton，macOS 不支持）。**本计划锁定 MoGe-2**，不引入 MoGe-3，避免依赖与显存范围失控。

---

## 4. 可行性分析：替换映射

### 4.1 字段映射表

| 现有代码 | 位置 | MoGe-2 写法 |
| --- | --- | --- |
| `depth_pro.create_model_and_transforms(device=...)` | [L2339](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L2339) | `MoGeModel.from_pretrained("Ruicheng/moge-2-vitl").to(device).eval()` |
| `transform(rgb).to(device)` | [L2453](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L2453) | `torch.from_numpy(rgb).permute(2,0,1).float()/255`（MoGe 内部自行 resize，不需要外部 transform） |
| `pred["depth"].squeeze()` | [L2456](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L2456) | `out["depth"]` |
| `pred["focallength_px"]` | [L2457](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L2457) | `out["intrinsics"]`（归一化） |
| `K = [[f_px,0,W/2],[0,f_px,H/2],[0,0,1]]` | [L2463](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L2463) | `K = [[fx*W,0,cx*W],[0,fy*H,cy*H],[0,0,1]]` |

### 4.2 内参语义变化（最容易踩的坑）

现状：DepthPro 只给 `focallength_px`，主点**硬编码为图像中心** `(W/2, H/2)`。

MoGe-2：给完整归一化 K，主点**可能不在中心**（裁切/透视场估计的结果）。

影响面：

1. `extract_ground` → `depth_to_points` 用 `inv(K)` 反投影，主点偏移会让点云整体剪切，地面法向产生**系统性偏转**——这正是姿势世界系倾角（`tilt_deg`）的直接来源。
2. `_keypoints_depth_camera` 用 `cx, cy` 做反投影（[L353-L354](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L353-L354)），主点变化会平移关键点目标点。
3. `person_bbox_3d_m` / `contact_pixels` 同样受 `K` 影响。

结论：**必须用 MoGe 自己的 `cx/cy`，不要图省事回填 `W/2, H/2`**。若实测发现 MoGe 主点偏移较大（> 2% 边长），需要在回归里单独记录，因为它是"精度是否下降"的一个独立变量。

### 4.3 输出分辨率对齐

`extract_ground` 有硬校验：`depth.shape == image_rgb.shape[:2]`（[pipeline.py L57](file:///g:/UGit/Phy-SIC/ground_pipeline/pipeline.py#L55-L58)）。

- DepthPro 内部 resize 到 1536² 再把 depth 插值回原图（[depth_pro.py L271-L293](file:///g:/UGit/Phy-SIC/external/ml-depth-pro/src/depth_pro/depth_pro.py#L271-L293)）。
- MoGe-2 声明输出与输入同尺寸，但 `resolution_level` / `num_tokens` 会在内部降分辨率后再上采样。**P1 里必须加一条断言**：`out["depth"].shape == rgb.shape[:2]`，不等就插值回原图并记日志，不允许静默通过。

### 4.4 加载顺序与 WSL 稳定性（不能动）

现有注释明确写着：**深度模型必须先于 LangSAM 加载**，否则 SAM2 作为第一个 CUDA 模型时 GroundingDINO 前向会让 WSL VM 崩溃（[L2336-L2340](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L2336-L2341)）。

替换后必须保持同一顺序：`MoGe2 → LangSAM → SAM3D`。这是硬约束，P1 的适配层不能为了"懒加载"把这个顺序打乱。

### 4.5 代码改动清单

| 改动 | 文件 | 规模 |
| --- | --- | --- |
| 新增深度后端适配层（`depthpro` / `moge2` 两个实现，统一 `infer(rgb) -> (depth, K)`） | 新增 `ground_pipeline/depth_backend.py` | ~120 行 |
| 启动加载走适配层 | [ground_service.py L2334-L2343](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L2334-L2343) | 小 |
| 两处推理调用改走适配层 | [L2451-L2463](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L2451-L2463)、[L1597-L1606](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L1597-L1606) | 小 |
| `/health` 增加 `depth_backend` 字段 | [L2387](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L2387) | 小 |
| 新增 CLI：`--depth-backend`、`--moge-model`、`--moge-num-tokens`、`--depth-fp16` | [L2984 起](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L2984-L2991) | 小 |
| 注释/图例去 DepthPro 化（改为 "depth model" 或按后端取名） | L662、L838、L906、L1128、L1137、L1584、L1871、L2702 | 文案级 |
| 分段计时 | `_pose_infer_ground_transform` + `/v1/ground` | 小 |

**不需要改**：`pipeline.py`（`extract_ground` 接口不变）、SAM3D 相关逻辑、重定向逻辑、响应 schema。

---

## 5. 「能否加速」的判定框架

### 5.1 前置：先量出各段占比（P0）

在 `/v1/ground` 与 `/v1/pose/single` 里加 5 个计时点，输出到响应与日志：

| 计时点 | 覆盖内容 |
| --- | --- |
| `segment_ms` | LangSAM（GroundingDINO 检测 + SAM2 分割 + mask 后处理） |
| `depth_ms` | transform + 深度模型前向 + 转 numpy |
| `ground_ms` | `extract_ground`（RANSAC 256 次 + SVD） |
| `pose_ms` | SAM3D Body 前向 |
| `post_ms` | 落地/对齐/ICP/重定向/可视化编码 |

同时记录 `torch.cuda.max_memory_allocated()`。

**为什么必须先做**：如果 `depth_ms` 只占 `time_ms` 的 15%，即使 MoGe-2 快 5 倍，端到端也只快 12%，却要承担地平面重标定的风险——这时候应该优先优化 SAM3D 或 `extract_ground`（例如 `iterations=256` 的 RANSAC 在 2 K 分辨率下也不便宜）。

### 5.2 理论对比（不作为验收依据）

| 项 | DepthPro | MoGe-2 ViT-L | MoGe-2 ViT-B |
| --- | --- | --- | --- |
| 参数量 | ≈646.5 M | 326 M | 104 M |
| 输入 | 固定 1536×1536（2.25 MP） | 自适应（`num_tokens` 控制） | 自适应 |
| 官方速度 | ~300 ms（V100 量级，来源：第三方模型库） | ~60 ms（A100/3090，FP16） | README 未给，按参数量应更快 |
| 显存 | ≈2.5 GB | 未标称，**需实测** | 未标称，**需实测** |
| 额外输出 | — | metric points / intrinsics / valid mask | 同左 |

**所有速度数字均为官方/第三方声明，必须在 RTX 4070 12 GB 上重测。** 4070 与 3090 同代但显存带宽与 FP16 吞吐更低，实际值大概率高于 60 ms。

### 5.3 判定门槛

采用 MoGe-2 的**唯一硬条件**（P0 出数后判定）：

1. `depth_ms` 在 `/v1/ground` 中占比 ≥ 25%，或 `depth_ms` 绝对值 ≥ 150 ms；**且**
2. 本机实测 MoGe-2 的最快可用档位（ViT-L FP16 或 ViT-B FP16）比 DepthPro 快 ≥ 40%；**且**
3. §6 的精度回归全部通过。

三条不同时满足 → **不换**，把 P1 预算转投 SAM3D 与分割阶段的优化。

### 5.4 若加速不足，可叠加的手段（按性价比排序）

| 手段 | 预期 | 风险 |
| --- | --- | --- |
| FP16（`use_fp16`） | 中 | 深度数值精度，需回归 |
| `num_tokens` 下调至 1200–1500 | 中 | 细节变钝 → 地面边缘噪声 |
| 换 ViT-B | 高 | 米制尺度精度未知，需专门验证 |
| 降 `extract_ground` 的 `iterations` | 低-中 | 平面稳定性 |

---

## 6. 「是否减姿势精度」的判定框架

### 6.1 深度在姿势精度里的作用链

```
depth + K ──► 1) 地平面 (normal, d) ──► 世界变换 R ──► 相机高度、落地、穿地率
          └──► 2) 关键点处深度采样 ──► 目标点 target ──► 残差诊断 / 回退 Umeyama
```

### 6.2 主路径为什么不依赖深度尺度

`_align_pose_mesh_to_depth` 的主路径 `camera_ground_anchor` 明确写死 `scale = 1.0`（[L848-L876](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L848-L876)）。设计意图写在注释里：SAM3D 在相机空间本来就站在自己感知的地面上，深度地面的作用是**确定世界系朝向与高度**，不是缩放网格。

因此：

- **深度模型的绝对尺度误差（比如整体偏 5%）不会让网格缩水** —— 这是历史上 `scale=0.7878` 事故之后刻意做的隔离。
- 深度真正影响的是：地平面的**倾角**与**高度** → 世界系 `tilt_deg`、`shift_m`、接触高度、穿地比例。

这就把"精度下降"的风险从"全局尺度"收窄到"地平面拟合质量"，是可测量、可标定的。

### 6.3 三处必须重标定的耦合点

| # | 耦合点 | 现状 | 为什么必须重标定 |
| --- | --- | --- | --- |
| 1 | `extract_ground(distance_threshold=0.02)` | 绝对 2 cm 内点阈值（[pipeline.py L32](file:///g:/UGit/Phy-SIC/ground_pipeline/pipeline.py#L26-L37)） | 该阈值是按 DepthPro 在 1536²→原图的深度噪声分布调出来的。MoGe-2 若边界更锐利但远处噪声更大，2 cm 可能过松（吸进非地面）或过紧（内点不足→平面抖动） |
| 2 | `_keypoints_depth_camera` 的 `spread / 0.04` | 4 px 邻域 MAD 的经验尺度 4 cm（[L374-L377](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L374-L377)） | 权重公式对噪声尺度敏感。深度噪声变小会让所有关键点权重趋近 1，等于取消鲁棒性；变大会让权重趋近 0，等于禁用回退路径 |
| 3 | 回退路径 Umeyama 的 `0.5 <= scale <= 2.0` 接受区间 | [L893](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L893) | 回退路径会把深度尺度误差**吸收成网格缩放**（历史事故根因）。MoGe-2 若尺度误差方向/幅度不同，这个区间可能过宽，需要按实测收紧 |

### 6.4 精度基线（来自既有验收记录）

以 [验收失败记录_SAM3D姿势优化.md](file:///g:/UGit/Phy-SIC/ground_pipeline/验收失败记录_SAM3D姿势优化.md#L165-L180) 的 `camera_ground_anchor` 方案为当前基线：

| 指标 | `5306c1d3...jpg` | `yoga_core.jpg` |
| --- | --- | --- |
| `rmse_m` | 0.0956 | 0.3459 |
| `rmse_final_m` | 0.0840 | 0.3449 |
| `ground_anchor.tilt_deg` | 3.47° | 1.79° |
| `ground_anchor.shift_m` | +8.99 cm | +2.50 cm |
| Chamfer | 0.00236 | 0.00961 |

`yoga_core.jpg` 的 `rmse_m = 0.3459` 已归因为 **DepthPro 深度尺度偏差**（2D 重投影只有约 3 cm）。这条尤其值得关注：**如果 MoGe-2 的米制尺度更准，yoga 这张图的指标应当变好而不是变差**——它是本次替换最灵敏的探针图。

### 6.5 验收判定（任一不满足即判定"减精度"，回滚）

| 指标 | 允许变化 |
| --- | --- |
| `rmse_m` | 两张图都不劣化超过 **+10%**，且 `5306` 不劣化超过 +5% |
| `rmse_final_m` | 同上 |
| `ground_anchor.tilt_deg` | 不超过 **4.5°**（当前最差 3.47°） |
| `ground_anchor.shift_m` | 绝对值不超过 **12 cm**（当前最差 8.99 cm） |
| 接触高度 | 绝对值不超过 **3 cm** |
| 穿地比例 | 不超过 **5%** |
| `status` | 必须仍是 `success` 且 `method == "camera_ground_anchor"`（**不允许退化到回退路径**） |
| 平面内点率 | `plane_inlier_mask.sum() / (floor_mask & valid).sum()` 不低于基线的 **0.85×** |

第七条是最重要的守门条件：一旦回退到 `weighted_keypoint_umeyama`，尺度的历史事故就会复活。

---

## 7. 实施方案

### P0 —— 只measure，不改模型（必须先做）

1. `/v1/ground` 与 `/v1/pose/single` 加 5 段计时 + `max_memory_allocated`，写进响应与日志。
2. 新增 `ground_pipeline/bench_depth.py`（或临时脚本）：对三张基线图，跑 DepthPro / MoGe-2 ViT-L(FP32|FP16) / MoGe-2 ViT-B(FP16)，输出 `depth_ms`、显存峰值、`f_px` 与 `depth` 统计（中位数、P5/P95、`depth_range_m`）。
3. 产出表格：**深度阶段占比** + **MoGe-2 vs DepthPro 实测加速比**。
4. 按 §5.3 判定，出 go / no-go 结论。

> P0 交付物是一张表，不是代码。若 no-go，本计划到此结束。

### P1 —— 适配层 + 开关 + 回滚（go 才做）

1. 新增 `depth_backend.py`：`class DepthBackend(Protocol): infer(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]`，两个实现 `DepthProBackend` / `MoGe2Backend`。约定：返回的 `depth` 必须是**原图分辨率米制**，`K` 必须是**像素 3×3**。
2. `MoGe2Backend` 内做三件事：归一化 K → 像素 K；断言/插值对齐 `depth.shape == rgb.shape[:2]`；`valid_mask` 仅用于日志统计，不改变下游行为。
3. 启动加载改走适配层，**保持「深度模型先于 LangSAM」顺序**。
4. CLI：`--depth-backend {depthpro,moge2}`（默认 `depthpro`）、`--moge-model`（默认 `Ruicheng/moge-2-vitl`）、`--moge-num-tokens`（默认不传，用库默认）、`--depth-fp16`（默认关）。
5. `/health` 返回 `depth_backend` 与模型名，便于灰度期确认实例身份。
6. **回滚开关就是 `--depth-backend depthpro`**，无需回滚代码。

### P2 —— 标定与回归

1. 用 P0 的 `depth` 统计对比两个后端在**地面掩码内**的噪声分布（MAD、P95-P5），据此重标 §6.3 的三处常数。
2. 三图回归，逐项对 §6.5 阈值。
3. 显存叠加测试：MoGe-2 + LangSAM + SAM3D 三者共存，确认 12 GB 内不 OOM；对比 DepthPro 场景的峰值。
4. 并发场景：单图 × 连续 20 次，看是否有 `max_memory_allocated` 爬升（MoGe 若内部缓存 token grid 需注意）。

### P3 —— 灰度与默认切换

1. 先以 `--depth-backend moge2` 在单个实例上跑一周，观察 `/metrics` 中的耗时与错误率。
2. 无异常后把默认值改为 `moge2`，`depthpro` 保留两个版本作为回滚路径。
3. 两个版本之后可删 DepthPro 相关代码与 ASCL 依赖（需另开清理任务）。

---

## 8. 测试与验收清单

| # | 项 | 通过标准 |
| --- | --- | --- |
| 1 | 单图 `depth_ms` | MoGe-2 < DepthPro，且达到 §5.3 门槛 |
| 2 | `depth.shape` 对齐 | 三图 × 两后端，全部等于原图 H×W |
| 3 | K 语义 | MoGe 的像素 K 与归一化 K 互逆一致（`fx_norm * W == fx_px`） |
| 4 | 地平面稳定性 | 同图重复 5 次，`plane_normal` 夹角 < 0.2°（`random_seed=0` 已固定） |
| 5 | 三图精度回归 | 全部满足 §6.5 |
| 6 | 回退路径未触发 | `method` 恒为 `camera_ground_anchor` |
| 7 | 显存 | 峰值 < 11 GB（留 1 GB 余量），无 OOM |
| 8 | WSL 稳定性 | 连续 50 次请求不崩 VM |
| 9 | 回滚 | `--depth-backend depthpro` 启动后行为与当前版本逐字段一致 |

---

## 9. 风险与权衡

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| **深度不是瓶颈** | 换了模型加速有限，却引入标定风险 | P0 先判定，占比 < 25% 直接 no-go |
| **MoGe-2 米制尺度精度未知** | 地平面高度偏移 → 世界系整体漂移 | 三图回归 + `yoga` 探针图 + §6.5 门槛 |
| **主点不再居中** | 点云剪切 → 地面倾角系统性偏差 | 使用 MoGe 自己的 `cx/cy`；回归中单列该指标 |
| **`extract_ground` 2 cm 阈值失配** | 内点过少 → 平面抖动 / 拟合失败 | P2 重标定 + `plane_inlier` 比率门槛 |
| **`spread/0.04` 权重失配** | 回退路径被意外激活或禁用 | P2 重标定；回归中监控 `method` |
| **新依赖 + numpy 版本冲突** | `ground-wsl` 装不上（DepthPro 侧要求 `numpy<2`） | P1 前先在隔离环境做依赖试装；MoGe 依赖 `utils3d`、`trimesh` 需一并核对 |
| **`moge.model.v2` 是否在已 pin 的 commit 内** | 需升级依赖，可能影响全量流程的 `moge.model.v1` 导入 | 单独验证；v1/v2 模块可共存，但必须实测 |
| **WSL 加载顺序被打乱** | VM 崩溃（历史已知问题） | 适配层保持 `depth → LangSAM → SAM3D` |
| **MoGe-3 范围蔓延** | 1.25 B 参数、Triton 依赖，12 GB 显存吃紧 | 本文明确锁定 MoGe-2 |
| **FP16 数值问题** | 近距离地面深度抖动 | 默认 FP32；FP16 需单独回归通过才允许开 |

---

## 10. 待确认事项

| # | 事项 | 需要谁定 |
| --- | --- | --- |
| 1 | 是否允许在 `ground-wsl` 环境引入新依赖（`moge` + `utils3d` + `trimesh`） | 用户 |
| 2 | 目标 GPU 是否就是当前实测的 RTX 4070 12 GB（还是另有部署机） | 用户 |
| 3 | 是否接受"P0 只出数据、可能 no-go"的推进方式（即允许这项调研以"不换"结案） | 用户 |
| 4 | 是否把「DepthPro 非商业许可 → MIT」作为独立收益计入决策（即使加速不明显） | 用户 |
| 5 | `yoga_core.jpg` 的 0.3459 m 基线是否仍作为验收探针（它同时受 SAM3D 本身误差影响） | 用户 |

---

## 11. 本计划明确不做

- 不替换 LangSAM（分割质量与地面 mask 直接耦合，属另一立项）
- 不使用 MoGe-2 的 `normal` 输出做地平面拟合（有新方法风险，且 `extract_ground` 现有 RANSAC 路径已验证）
- 不引入 MoGe-3
- 不改动 `pipeline.py::extract_ground` 的算法结构（只允许改默认阈值）
- 不改动 SAM3D / 落地 / 重定向逻辑
- 不在 P0 阶段写任何模型替换代码

---

## 附：关键文件索引

| 文件 | 作用 |
| --- | --- |
| [ground_service.py](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py) | 服务主体；深度 4 处消费点、加载顺序、姿势对齐 |
| [pipeline.py](file:///g:/UGit/Phy-SIC/ground_pipeline/pipeline.py) | `extract_ground` RANSAC/SVD，2 cm 阈值所在 |
| [utils/depth.py](file:///g:/UGit/Phy-SIC/utils/depth.py) | 全量流程的 MoGe v1 + DepthPro 对齐历史实现，可参考 API 用法 |
| [ml-depth-pro/depth_pro.py](file:///g:/UGit/Phy-SIC/external/ml-depth-pro/src/depth_pro/depth_pro.py#L244-L298) | DepthPro `infer` 的 resize/输出语义 |
| [WSL_SETUP.md](file:///g:/UGit/Phy-SIC/ground_pipeline/WSL_SETUP.md) | 轻量服务环境（Python 3.10 / torch 2.5.1 / cu121） |
| [验收失败记录_SAM3D姿势优化.md](file:///g:/UGit/Phy-SIC/ground_pipeline/验收失败记录_SAM3D姿势优化.md) | 精度基线与指标定义 |
| [计划_姿势推理服务队列_20261026.md](file:///g:/UGit/Phy-SIC/ground_pipeline/计划_姿势推理服务队列_20261026.md) | 并行推进的队列改造；深度替换的计时改造应与其 `/metrics` 对齐 |
