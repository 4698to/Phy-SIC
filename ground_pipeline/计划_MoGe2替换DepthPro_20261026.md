# 计划：用 MoGe-2 替换 DepthPro（加速与姿势精度评估）

- 日期：2026-10-26
- 状态：**P0/P1 完成（2026-10-27）；性能门槛 1、2 通过；P2 地面重复性通过，但姿势精度门槛失败；当前结论：不切换默认后端**
- 范围：`ground_pipeline` 地面/姿势推理服务中的**深度模型**替换，不含分割模型（LangSAM）、姿势模型（SAM3D Body）替换
- 结论先行见 §0；基准及回归结果见 §5.1.4、§5.3、§6.5

---

## 前置：当前 ground + pose 推理流程与模块职责

> 先把这条链路读清楚——DepthPro 只是其中一段，本计划讨论的就是替换它。

### F.1 两条入口的完整流程

> 下图标注 DepthPro，表示本计划实施前的原始流程；改造后深度模型由启动参数选择，默认仍为 DepthPro，候选为 MoGe-2。

**① `/v1/pose/single`（`/v2/pose/*` 同源，JSON 或 XML）**

```
请求线程（CPU，不入队）              GPU worker（PoseQueue 单线程，串行）
────────────────────────────        ─────────────────────────────────────────────
_pose_prepare                       _pose_execute
 · 参数校验 / 读图 → rgb      ──►    1. _person_mask   LangSAM("person.")
 · resolve_target（目标骨架）  入队   2. _pose_resolve_ground
                               +       ├ ground_mode=request → 用请求给的平面
                               等待    └ auto/infer → _pose_infer_ground_transform
                                          LangSAM("floor")  → floor_mask
                                          DepthPro(1536²)   → depth, f_px
                                          K = [f,0,W/2; 0,f,H/2; 0,0,1]
                                          extract_ground    → n, d（RANSAC+SVD）
                                       3. estimator.infer  SAM3D Body
                                          → SMPL-24 关节 / MHR 网格 / 2D-3D 关键点
                                       4. _pose_apply_landing   消穿地，贴到地面
                                       5. assemble_pose + retarget（figure_v6）
                                       6. _pose_resolve_world   → ground_world_v1
                                       7. gravity / overlay 编码 / timings
```

**② `/v1/ground`**

```
LangSAM("floor")   → floor_mask
LangSAM("person.") → person_mask（只用于 bbox / 接触区，失败不影响地面）
DepthPro(1536²)    → depth, f_px → K
extract_ground     → points（相机系点云）, plane_normal, plane_d, inlier_mask
├─ person_bbox_3d_m / person_centroid_m / contact_pixels（都用 points 反查）
├─ [pose_inference=true] → 同一个 PoseQueue（低优先级 PRIORITY_GROUND）
│     只跑 SAM3D Body（复用本请求算好的 K），不带落地/重定向/world；
│     队列满/超时只把 pose 降级为 error，不影响地面结果
└─ visualize=true → 可视化；若 pose 成功且要 mesh，再跑
      _align_pose_mesh_to_depth（camera_ground_anchor 主路径 / Umeyama+ICP 回退）
```

两条入口共用**同一个 `PoseQueue` 单 worker**（GPU 串行），因此 `/v1/ground` 的姿势分支与 `/v1/pose/*` 会互相排队；`/v1/ground` 的「分割 + 深度 + 地面」三段不经过队列，直接跑在请求协程里。

### F.2 各模块在处理什么

| 环节 | 实现位置 | 输入 → 输出 | 实际在算什么 |
| --- | --- | --- | --- |
| **LangSAM**（GroundingDINO + SAM2.1） | `lang_sam.LangSAM` | RGB + 文本提示 → 布尔掩码 | 开放词表检测出框，再由 SAM2.1 分割。纯 2D 像素操作，不产生任何几何量。`floor_prompt`（默认 `floor.`）与 `person_prompt`（默认 `person.`）各调一次 |
| **DepthPro** | `depth_pro.create_model_and_transforms` + `model.infer`（[depth_pro.py](file:///g:/UGit/Phy-SIC/external/ml-depth-pro/src/depth_pro/depth_pro.py#L244-L298)） | RGB → `depth`（米，原图 H×W）+ `focallength_px` | 单目米制深度。内部固定 resize 到 **1536×1536** 推理，再把 depth 插值回原图；**只给焦距，不给主点** |
| **内参构造** | [ground_service.py L1646-L1649](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L1646-L1649) | `f_px`, W, H → 3×3 K | 主点硬编码为图像中心 `(W/2, H/2)`，`fy = fx`。这是 §4.2 说的替换风险点之一 |
| **extract_ground** | [pipeline.py L26-L138](file:///g:/UGit/Phy-SIC/ground_pipeline/pipeline.py#L26-L138) | depth + K + floor_mask → 点云 / 平面 / 内点 / 网格 | `depth_to_points` 逐像素反投影（`K⁻¹`）；对 floor_mask 内点做 **256 次 RANSAC**（距离阈 2 cm，且要求 `|n_y| ≥ 0.5` 剔除墙面）→ 内点 SVD 精化 → `plane_normal`、`plane_d`（`n·p + d = 0`）+ 平面内采样网格。**纯 CPU numpy**，大图上是主要耗时之一 |
| **GroundTransform** | [ground_retarget.py L350-L420](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_retarget.py#L350-L420) | 平面 `n, d` → 相机→世界刚体变换 | 用 Rodrigues 把地面法向对齐到世界 **+Z**（地面即 z=0，z 朝上）；`translation = (0, 0, -d)`，`camera_height_m = |d|` |
| **SAM3D Body** | [ground_pose.py L234-L311](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_pose.py#L234-L311) | RGB + bbox（+ 可选 K）→ 网格 / 关节 / 关键点 / focal / cam_t | 人体重建。`/v1/pose/single` 传 `cam_int=None`（用模型自己的默认 FOV 估焦距）；`/v1/ground` 分支把 DepthPro 的 K 传进去。可选 `SAM3D_UPSCALE` 放大推理再还原 2D 坐标 |
| **落地 landing_shift** | [ground_retarget.py L1026](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_retarget.py#L1026) | 世界系网格最低点 vs 地面 → 位移 | 按地面法向把网格「最低分位点」贴到 z=0，消掉穿地/悬浮，不改变体型与姿态 |
| **网格对齐（仅可视化）** | [ground_service.py L806-L906](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L806-L906) | 70 个关键点 + 深度 → 对齐后的网格 | 主路径 `camera_ground_anchor`：**scale 恒为 1.0**，只做旋转+平移；回退路径才用加权 Umeyama（会吸收尺度误差）+ trimmed ICP。这正是「主路径不依赖深度尺度」的地方 |
| **重定向 RetargetAdapter** | [ground_retarget.py L1310 起](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_retarget.py#L1310-L1485) | SAM3D 原始输出 → `generic` 骨架 → 目标骨架 | 归一到统一骨骼语义，再映射到目标骨架（figure_v6 等），输出 XML/JSON 两套序列化。与深度无关 |
| **PoseQueue** | [pose_queue.py](file:///g:/UGit/Phy-SIC/ground_pipeline/pose_queue.py) | 请求 → 单 worker 串行执行 | 单 GPU 不能并行跑 SAM3D，所以用容量 + 每客户端配额 + 老化（aging）+ 超时来排队，而不是让请求同时压进 GPU |

### F.3 单位与坐标系约定

- `depth`：米；`points`：**相机系**，x 右、y 下、z 前（与 OpenCV 一致）。
- 平面：`|n| = 1`，满足 `n·p + d = 0`；`d` 的绝对值即相机到地面的垂直距离（米）。
- 世界系 `ground_world_v1`：由 [GroundTransform](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_retarget.py#L350-L394) 定义，地面为 `z = 0`、`+z` 朝上，`rotation`/`translation` 可直接用于任何相机系点。
- 深度模型在整个链路里只贡献两样东西：**`depth` 数组** 和 **`focallength_px`**。只要替换后仍输出「原图分辨率米制 depth + 像素 K」，下游（点云、接触区、地平面、world、可视化）全部不用改。

---

## 0. 结论摘要（TL;DR）

| 问题 | 结论 | 依据强度 |
| --- | --- | --- |
| MoGe-2 能否在能力上替代 DepthPro？ | **能**。MoGe-2 一次前向同时给出米制深度 + 归一化相机内参 + valid mask，接口上覆盖 DepthPro 的 `depth` + `focallength_px` 两项输出 | 官方 README / 论文摘要，已验证 |
| 能否加速？ | **实测显著加速**。RTX 4070 / WSL2 三图基准：DepthPro 中位数 **1.67–1.81 s**；MoGe-2 ViT-L FP16 **133–210 ms**（快 **88.7–92.2%**），ViT-L FP32 **394–478 ms**（快 **74.2–77.1%**），ViT-B-normal FP16 **88–178 ms**（快 **90.2–94.7%**）。MoGe 权重峰值额外显存约 1.6–3.4 GB。详见 §5.1.4 | **本机实测**（2026-10-27，单模型基准） |
| 会不会掉姿势精度？ | **当前不满足切换条件**。qco6j 的 `camera_ground_anchor` / `scale=1.0` 主路径仍成功，但 `rmse_m=0.3871`（基线 0.0956）显著超出 +5% 门槛；接触高度 2.85 cm 在 3 cm 限值内。姿势精度门槛失败，需先查明地面/深度/对齐差异并重新标定 | **本机实测**（2026-10-27，单图网格探针；非受控 DepthPro 同机对照） |
| 建议动作 | **保留 DepthPro 为默认；MoGe-2 仅作为显式候选后端。** 性能门槛 1、2 已通过，但 P2 姿势精度门槛失败，暂停 P3 灰度与默认切换；地面重复性通过不代表姿势精度通过 | 本计划 + 实测 |

一句话：**深度段是 1024 px 图的第一大耗时（55%），但 24 MP 图主要耗在 `extract_ground`；MoGe-2 单模型虽显著提速，姿势精度硬门槛未过，因此目前只保留为显式候选，不推进灰度或默认切换。**

---

## 1. 背景与现状

### 1.1 服务链路

`ground_service.py` 单进程 FastAPI + 单张 GPU，模型串行（两条入口的完整流程与模块分工见文首「前置：当前 ground + pose 推理流程与模块职责」）：

```
RGB ──► LangSAM(groundingdino+sam2.1) ──► floor_mask
    └──► 选定深度后端（默认 DepthPro；候选 MoGe-2） ──► 原图分辨率米制 depth + 像素 K
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

### 1.3 现有耗时观测能力（P0 已闭合）

**改造前**全文件只有两处计时：

- `/v1/ground`：`time_ms` = 整个请求总耗时（[L2279](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L2279)）
- `/v1|v2/pose/single`：`pose["time_ms"]` 只覆盖 SAM3D 那一段（[L2485-L2493](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L2485-L2493)）

**改造后（P0，2026-10-27）**：`/v1/ground` 与 `/v1/pose/single` 均返回 `timings` 字典，并写一行 `stage_timing endpoint=... key=value ...` 日志（便于脚本解析）：

| 字段 | 语义 | `/v1/ground` | `/v1/pose/single` |
| --- | --- | --- | --- |
| `segment_ms` | 人物/地面 LangSAM 分割 | ✓ | ✓ |
| `person_segment_ms` | 人物 mask（仅姿势链路） | — | ✓ |
| `depth_ms` | 所选深度后端预处理 + 前向 + 转 numpy | ✓ | ✓ |
| `depth_focal_px` | 所选深度后端提供/换算出的焦距（MoGe-2 同时提供完整像素内参 K） | ✓ | ✓ |
| `ground_ms` | `extract_ground`（RANSAC + SVD） | ✓ | ✓ |
| `pose_ms` | SAM3D Body 前向 | ✓（`pose_inference` 分支） | ✓ |
| `post_ms` | 余量：落地/对齐/ICP/重定向/world/base64/overlay 编码 | ✓ | ✓ |
| `total_ms` / `exec_total_ms` | 服务端端到端 | ✓ | ✓ |
| `gpu_alloc_*` / `gpu_peak_mb` | `memory_allocated` / `max_memory_allocated` | ✓ | ✓ |

`timings` 只在 JSON 响应里出现，v2 XML 输出会把它剥离（与 `queue` / `error_code` / `retry_after_s` 同处理），XML 结构不变。

⚠️ `post_ms` 是**减法余量**，不是独立打点：它包含 SAM3D 内部的前后处理、世界系组装、base64/PNG 编码与响应组装。因此它只用于判断「深度之外还有什么在花时间」，不能当作某一段的精确耗时。


### 1.4 仓库里已有的 MoGe 痕迹（重要）

MoGe 不是陌生依赖，本仓库全量流程已经在用：

| 位置 | 内容 |
| --- | --- |
| [requirements.txt](file:///g:/UGit/Phy-SIC/requirements.txt#L12) | 已 pin `git+https://github.com/microsoft/MoGe.git@0286b49...` |
| [utils/depth.py](file:///g:/UGit/Phy-SIC/utils/depth.py#L18-L34) | `from moge.model.v1 import MoGeModel`；`MoGeModel.from_pretrained("Ruicheng/moge-vitl")` |
| [utils/depth.py](file:///g:/UGit/Phy-SIC/utils/depth.py#L107-L124) | `run_metric_moge_from_depthpro`：**用 DepthPro 把 MoGe 的 depth 对齐成米制** |
| [utils/depth.py](file:///g:/UGit/Phy-SIC/utils/depth.py#L156-L163) | 注释：`This is only required for MoGe v1: v2 returns metric points.` |
| [cfg/v1.yaml](file:///g:/UGit/Phy-SIC/cfg/v1.yaml#L40) | `f_px: moge` —— 全量流程用 MoGe 的内参选支 |

1. **MoGe-2 已安装并验证**：目标服务环境 `/mnt/g/UGit/qwen3-vl/.venv-ground` 已安装 MoGe 2.0.0 与依赖，`moge.model.v2` 可导入；权重通过 `https://hf-mirror.com` 获取并完成推理验证。仓库依赖 pin 与目标环境实际模块版本需分别看待。
2. **保留既有 MoGe v1 集成**：`utils/depth.py` 继续使用 `moge.model.v1`；服务适配层新增 MoGe-2 v2 后端，不替换全量流程中的 v1。

---

## 2. 待回答的问题

| 编号 | 问题 | 回答方式 |
| --- | --- | --- |
| Q1 | 深度阶段在 `/v1/ground` 与 `/v1/pose/single` 中各占多少毫秒/百分比？ | ✅ **已答**（§5.1.1）：1.85–2.07 s；1024 px 图 55–56% / 33–40%，24 MP 图 12–14% |
| Q2 | 在目标 GPU（RTX 4070 12 GB）上，MoGe-2 ViT-L / ViT-B 实测多少毫秒、多少显存？ | ✅ **已答**（§5.1.4）：ViT-L FP16/FP32 与 ViT-B-normal FP16 的耗时、峰值分配显存已实测 |
| Q3 | MoGe-2 的米制深度的绝对尺度误差是多少？与 DepthPro 相比在同一张图上差多少？ | ⏸️ 未完成：没有受控同服务深度 A/B 或已知尺度真值 |
| Q4 | 换模型后地平面法向/高度漂移多少？姿势指标是否劣化？ | 🟡 部分回答：MoGe 地面输出与 qco6j 姿势探针见 §5.1.5、§6.5；无受控 DepthPro 平面对照，姿势 RMSE 超限 |
| Q5 | 显存叠加：DepthPro(≈2.5 GB) 换成 MoGe-2 后，与 SAM3D ViT-H 共存是否更宽裕？ | ⏸️ 未完成：未做独占 GPU 的整服务显存对照 |

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
| 本项目环境 | 服务环境 `/mnt/g/UGit/qwen3-vl/.venv-ground` 为 Python 3.10（依赖安装与 `moge.model.v2` 导入已验证）；`moge` v2 与 `utils3d` 已安装 | 已安装并 `pip install -e` |

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
| 分段计时 | `_pose_infer_ground_transform` + `/v1/ground` | ✅ 已完成（§1.3，未提交） |

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

### 5.1.1 实测结果（P0，2026-10-27 完成）

环境：RTX 4070 Laptop 12 GB（12282 MiB）、WSL2、`run_ground_pipeline.bat` 启动的真实服务（端口 8001）、**FP32**、DepthPro 固定 1536×1536 输入。三张基线图：

| 代号 | 文件 | 分辨率 |
| --- | --- | --- |
| `qco6j` | `5306c1d3gy1hsktgsqco6j20sg11wq4s.jpg` | 1024×1364 |
| `sz459j` | `5306c1d3gy1hsktgsz459j20sg0ugmyv.jpg` | 1024×1096 |
| `yoga` | `yoga_core.jpg` | 6000×4000（24 MP） |

单位为毫秒，取快样本中位数（剔除规则见 §5.1.2）：

| 图片 | 场景 | `segment` | **`depth`** | `ground` | `pose` | `post` | `total` | **depth 占比** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qco6j | `/v1/ground` | 606 | **1858** | 730 | — | 109 | 3301 | **56%** |
| qco6j | `/v1/ground` + `pose_inference` | 759 | **1882** | 673 | 1641 | 6454 | 11411 | 16% |
| qco6j | `/v1/pose/single` | 300 | **1849** | 670 | 1559 | 164 | 5687 | 33% |
| sz459j | `/v1/ground` | 780 | **1852** | 629 | — | 96 | 3369 | **55%** |
| sz459j | `/v1/ground` + `pose_inference` | 4040 | **3597** | 396 | 1886 | 6144 | 16064 | 22%（慢速边缘，n=1） |
| sz459j | `/v1/pose/single` | 259 | **1883** | 525 | 1570 | 155 | 4654 | 40% |
| yoga | `/v1/ground` | 1297 | **1927** | 7749 | — | 5094 | 16082 | 12% |
| yoga | `/v1/ground` + `pose_inference` | 1534 | **1994** | 7227 | 2563 | 16338 | 29658 | 7%（n=1） |
| yoga | `/v1/pose/single` | 548 | **2063** | 8892 | 2759 | 314 | 15100 | 14% |

读法与结论：

1. **`depth_ms` 与输入分辨率无关**：三张图（1.1 MP / 1.4 MP / 24 MP）都落在 **1.85–2.07 s**，因为 DepthPro 输入被固定到 1536²，原图尺寸只影响 transform 与插值回来的部分。
2. **1024 px 图上深度就是第一大头**：`/v1/ground` 占 **55–56%**，`/v1/pose/single` 占 **33–40%**（姿势链路里 SAM3D 约 1.6 s，与深度同量级）。
3. **24 MP 图上深度被挤到 12–14%**：`extract_ground` 飙到 **7.5–8.9 s**（RANSAC 256 次 + 每像素投影），`post` 也到 5.1 s / 16.3 s。这类图换深度模型几乎没用。
4. `post_ms` 在 `pose_inference` 分支上很大（6.1–16.3 s），主要来自 SAM3D 前后处理、世界系组装与 base64/PNG 编码——这解释了为什么 `/v1/ground` + pose 的 `total` 远大于两段直加。

性能实测：MoGe-2 ViT-B-normal FP16 单模型 `depth_ms` 约 88–178 ms，对比 DepthPro 1667–1814 ms，快约 90.2–94.7%；该值不等于端到端服务收益，见 §5.1.4/§5.1.5。

**端到端收益仅为按单模型实测比例折算的算术估算，不是服务回归结果**（忽略并发、显存压力与分段波动；真实 `/v1/ground` 与 pose 服务耗时见 §5.1.5）：

| 场景 | DepthPro 当前 total | 按单模型深度耗时差估算可省 | 粗略换后 total | 粗略端到端降幅 |
| --- | --- | --- | --- | --- |
| 1024 px `/v1/ground` | 3.30 s | 约 1.67 s | 约 1.63 s | 约 51% |
| 1024 px `/v1/pose/single` | 5.69 s | 约 1.67 s | 约 4.02 s | 约 29% |
| 24 MP `/v1/ground` | 16.08 s | 约 1.73 s | 约 14.35 s | 约 11% |

此估算只用于解释瓶颈占比；MoGe 服务测量受到系统负载影响，不能替代稳定、同机对照的端到端 A/B 验收。

### 5.1.2 测量口径与数据处理

- **采样设计**：三图 × 三种场景（`ground` / `ground+pose` / `pose_single`）**交错采集**，共 21 条记录；每格 2–3 次取中位数。交错是为了避免「慢速模式」（§5.1.3）整段污染某一场景，导致横向比较失真。
- **慢速样本剔除**：`depth_ms > 5000` 视为落入慢速模式，先剔除再取中位数；因此表中数字代表**显存健康状态下**的性能。
- **`total_ms` 口径**：服务端 `time_ms`，不含客户端网络与排队等待（队列为空时的单请求）。`post_ms` 为减法余量（见 §1.3 的警示）。
初始 P0 测量阶段尚未跑 FP16 / MoGe-2 / `num_tokens` 调参；这些后续在 P1/P2 完成的结果分别记录于 §5.1.4、§5.1.5。
- 表内 `sz459j` 与 `yoga` 的 `ground+pose` 各只有 1 个有效样本，仅作趋势参考，不作为门槛判定依据。

### 5.1.3 显存与「慢速模式」观测（重要）

| 指标 | 观测值 |
| --- | --- |
| `torch.cuda.max_memory_allocated`（进程峰值） | **9589.2 MB** |
| `nvidia-smi` 显存峰值（压测期间） | **11.7 / 12.3 GB** |
| 空闲回落 | 6270 MiB |
| `depth_alloc_delta_mb`（深度段前后 `memory_allocated` 差） | 17–21 MB（1024 px）/ 366 MB（24 MP，含点云） |

两点必须记住：

1. **`depth_alloc_delta_mb` 不能代表深度模型的真实占用**。PyTorch caching allocator 会复用已释放块，深度前向的临时张量多半命中缓存，所以差值只有几十 MB；真实占用来自进程常驻（LangSAM + SAM3D + DepthPro 权重与工作区），要看得看 `max_memory_allocated` 与 `nvidia-smi`。
2. **显存逼近上限会触发「慢速模式」**。压测期间多次出现 `depth_ms` 从 1.85 s 阶梯式劣化到 3.6 s / 11.6 s / 29.7 s / 41.7 s，此时 `nvidia-smi` 显示 GPU 利用率 100%、显存 11.7/12.3 GB；WSL 内无僵尸进程（只有 `ground_service` RSS ≈ 8.6 GB 与 Windows 侧常驻的 `clip_embedding.py`）。空闲 45 s 不恢复，停压数分钟后自动回到 1.87–1.94 s。判定为**显存打满后 WDDM 换页**，不是代码 bug，也不是模型本身变慢。

对后续阶段的影响：§8 第 7 条要求「峰值 < 11 GB、留 1 GB 余量」，而现状已经在 11.7 GB 徘徊——**P2 的显存叠加测试必须在独占 GPU 上进行**，否则测出来的既不是 MoGe-2 的显存，也不是真实性能。


### 5.1.4 MoGe-2 单模型基准（P1，2026-10-27 完成）

环境：RTX 4070 Laptop 12 GB、WSL2，服务环境 `/mnt/g/UGit/qwen3-vl/.venv-ground`；三张基线图，单模型推理基准，不含 LangSAM、`extract_ground`、SAM3D 或服务端到端开销。ViT-B 使用 `Ruicheng/moge-2-vitb-normal`。以下为各图重复运行的中位数范围；峰值显存为不同图片/运行间观测范围。

| 后端 / 精度 | `depth_ms` 中位数范围 | 峰值分配显存范围 | 相对 DepthPro 加速 |
| --- | ---: | ---: | ---: |
| DepthPro FP32 | 1667–1814 ms | 基准运行；模型与进程常驻显存口径不同 | — |
| MoGe-2 ViT-L FP16 | 133–210 ms | 2889–3375 MB | 88.7–92.2% |
| MoGe-2 ViT-L FP32 | 394–478 ms | 2238–2743 MB | 74.2–77.1% |
| MoGe-2 ViT-B-normal FP16 | 88–178 ms | 1632–2485 MB | 90.2–94.7% |

> 加速百分比按 DepthPro 与 MoGe 对应图像中位数计算；只代表单模型基准，不可外推为服务端端到端加速。MoGe FP16 通过 `infer(..., use_fp16=True)` 控制内部精度；服务 adapter、模型加载顺序及默认 DepthPro 回滚路径均已验证。性能门槛（至少快 40%）通过。

### 5.1.5 MoGe-2 服务地面回归与重复性（P2，2026-10-27）

MoGe-2：`Ruicheng/moge-2-vitb-normal`、FP16；真实服务实例已通过 `/health` 确认，模型在 LangSAM 前加载。

| 图片 | 法向量 | `plane_d` / 高度 (m) | floor 像素 | 内点数 | `depth_ms` | `ground_ms` | `total_ms` | 峰值分配显存 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qco6j | `[-0.01247, 0.96433, 0.26441]` | `-1.11365` / 1.11365 | 1,151,516 | 424,637 | 405.0 ms（首请求） | 836.8 ms | 3,625.8 ms | 2,802.8 MB |
| sz459j | `[-0.01087, 0.96821, 0.24991]` | `-0.97544` / 0.97544 | 877,159 | 280,341 | 89.7 ms | 632.4 ms | 2,017.8 ms | 2,802.8 MB |
| yoga | `[0.00324, 0.99796, -0.06373]` | `-0.32988` / 0.32988 | 5,449,867 | 5,448,857 | 151.4 ms | 8,460.1 ms | 14,063.2 ms | 3,651.9 MB |

qco6j 五次重复拟合完全一致（精确到上表输出精度）：法向量 `[-0.012469667, 0.964329124, 0.264412314]`、`plane_d=-1.113651867`、floor 像素 1,151,516、内点 424,637；预热后 `depth_ms=86.7–95.9 ms`。因此地面拟合重复性检查通过，但它不代表相对 DepthPro 的平面精度通过。当前没有同一服务状态下的受控 DepthPro 平面对照；MoGe 前向过程中出现过无效深度参与投影的 RuntimeWarning，需在后续诊断时一并核查。服务端数据包含不同图像尺寸及系统负载，特别是 yoga 的 `ground_ms`，不用于单模型性能门槛。

### 5.2 理论对比（不作为验收依据）

| 项 | DepthPro | MoGe-2 ViT-L | MoGe-2 ViT-B |
| --- | --- | --- | --- |
| 参数量 | ≈646.5 M | 326 M | 104 M |
| 输入 | 固定 1536×1536（2.25 MP） | 自适应（`num_tokens` 控制） | 自适应 |
| 官方速度 | ~300 ms（V100 量级，来源：第三方模型库） | ~60 ms（A100/3090，FP16） | README 未给，按参数量应更快 |
| 显存 | ≈2.5 GB（DepthPro 基线进程口径，非与 MoGe 单模型同口径） | 单模型峰值分配显存 2.2–3.3 GB（见 §5.1.4） | 单模型峰值分配显存 1.6–2.5 GB（见 §5.1.4） |
| 额外输出 | — | metric points / intrinsics / valid mask | 同左 |

**本表仅作模型规模与输入方式的理论对照，不作为验收依据。** 本机实测已在 §5.1.1、§5.1.3、§5.1.4 分别记录服务基线、服务显存观测与单模型基准；各项测量口径不同，不应直接混为同一显存对照。

### 5.3 判定门槛

采用 MoGe-2 的**唯一硬条件**（P0 出数后判定）：

1. `depth_ms` 在 `/v1/ground` 中占比 ≥ 25%，或 `depth_ms` 绝对值 ≥ 150 ms；**且**
2. 本机实测 MoGe-2 的最快可用档位（ViT-L FP16 或 ViT-B-normal FP16）比 DepthPro 快 ≥ 40%；**且**
3. §6 的精度回归全部通过。

三条不同时满足 → **不换**，把 P1 预算转投 SAM3D 与分割阶段的优化。

**实测判定（2026-10-27）**：

| 门槛 | 判定 | 依据 |
| --- | --- | --- |
| 1. `depth_ms` 占比 ≥ 25% 或绝对值 ≥ 150 ms | ✅ **满足** | DepthPro 绝对值约 1.85–2.07 s；1024 px `/v1/ground` 占 55–56% |
| 2. MoGe-2 最快可用档位比 DepthPro 快 ≥ 40% | ✅ **满足** | ViT-B-normal FP16 三图加速约 90.2–94.7%；见 §5.1.4 |
| 3. §6 精度回归全部通过 | ❌ **失败** | qco6j `rmse_m=0.3871`，基线 0.0956，超过 +5% 限值；见 §6.5 |

结论：**性能上有明显收益，但由于精度硬门槛失败，整体 no-go；DepthPro 保持默认，不启动 P3 灰度/切换。** qco6j 是首个姿势探针且已失败，因此未继续排队其余姿势图；地面稳定性测试单独完成并通过。


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
                                              （只服务 visualize 的网格对齐支路，见 §6.6）
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
| 2 | `_keypoints_depth_camera` 的 `spread / 0.04` | 4 px 邻域 MAD 的经验尺度 4 cm（[L402-L410](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L402-L410)） | 权重公式对噪声尺度敏感。深度噪声变小会让所有关键点权重趋近 1，等于取消鲁棒性；变大会让权重趋近 0，等于禁用回退路径。**该点只作用于可视化网格对齐支路（§6.6），不影响接口输出的姿势数值** |
| 3 | 回退路径 Umeyama 的 `0.5 <= scale <= 2.0` 接受区间 | [L908 起](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L908-L930) | 回退路径会把深度尺度误差**吸收成网格缩放**（历史事故根因）。MoGe-2 若尺度误差方向/幅度不同，这个区间可能过宽，需要按实测收紧。同样只属可视化支路（§6.6） |

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

### 6.5 验收判定（任一不满足即判定“减精度”，回滚）

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

**P2 实测（2026-10-27）：qco6j 姿势/网格探针未通过，结论 no-go。** MoGe-2 ViT-B-normal FP16 服务返回 `pose_status=success`、`alignment_status=success`、`method=camera_ground_anchor`、`scale=1.0`，但 `rmse_m=0.3871`、`rmse_final_m=0.3842`；对比基线 `0.0956` / `0.0840`，均严重超过 qco6j +5% 容差。其他观测为 `tilt=3.32°`、`shift=-9.67 cm`、anchor 接触高度 0、refine 接触高度 2.85 cm、穿地率 0%。虽然这些单项满足阈值（接触高度接近上限），RMSE 硬门槛失败即足以判定整体不通过。此次不是同一服务状态下的 DepthPro 受控对照，故记录为 MoGe 实测失败信号，不据此断言单一根因；其余姿势图未测。

第七条是最重要的守门条件：一旦回退到 `weighted_keypoint_umeyama`，尺度的历史事故就会复活。

### 6.6 可视化网格对齐是边缘功能（后续改为网格 ↔ 点云）

**触发条件很窄**：只有 `visualize=true` + `pose_want_mesh=true` + 姿势成功时才会执行 `_align_pose_mesh_to_depth`（[L2967-L2999](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L2967-L2999)）。`/v1/pose/*` 的世界系输出、`/v1/ground` 的地面结果都不依赖这条支路。

由此得到两个操作结论：

1. §6.3 的 #2（`spread/0.04`）、#3（Umeyama 尺度区间）即使失配，影响也只是「可视化网格贴得准不准」，不会改变接口输出的姿势数值。P2 可以把它们排在 #1（地面拟合阈值）之后重标，或直接跳过并记录为已知偏差。
2. 但 §6.4 / §6.5 的 `rmse_m`、`method`、`tilt_deg`、`shift_m` **恰恰是从这条支路的 `alignment` 报告里取出来的**。所以三图回归必须显式开 `visualize=true` + `pose_want_mesh=true`，否则这些验收数字根本不存在。这条支路是「边缘功能」，但暂时是唯一的量化探针。

**后续演进（不在本期）**：MoGe 的 `points` 可以直接作为配准目标，把对齐方式从「70 关键点 + 深度采样 + Umeyama」换成「SAM3D 网格 ↔ 点云」的 Chamfer / ICP 配准，从而彻底摆脱 `spread/0.04` 与 Umeyama 尺度区间这类与深度噪声绑定的经验常数。本期不做，原因有三：要重标配准阈值、要重跑三图回归；点图在模型输出分辨率上可靠，人体边缘与远处偏稀疏；是否优于关键点路径尚无实测。

---

## 7. 实施方案

### P0 —— 只 measure，不改模型（已完成）

1. ✅ `/v1/ground` 与 `/v1/pose/single` 分段计时和显存观测已完成，见 §1.3。
2. ✅ 已新增 `ground_pipeline/bench_depth.py` 并完成 DepthPro / MoGe-2 ViT-L(FP32|FP16) / ViT-B-normal(FP16) 三图单模型基准，结果见 §5.1.4。
3. ✅ DepthPro 服务基线及性能门槛判定已记录，见 §5.1.1、§5.3。

### P1 —— 适配层 + 开关 + 回滚（已完成，默认保持 DepthPro）

1. ✅ `depth_backend.py` 实现 `DepthProBackend` / `MoGe2Backend`，统一返回原图分辨率米制 depth 与像素坐标 3×3 K。
2. ✅ MoGe 归一化 K 转像素 K；深度尺寸不符时插值回原图；FP16 参数传入 MoGe 内部推理。
3. ✅ 服务启动经适配层加载深度后端，保持深度模型先于 LangSAM 的顺序；CLI、`/health` 身份字段和 `--depth-backend depthpro` 回滚均已验证。
4. ✅ 默认仍为 DepthPro；MoGe-2 必须显式以 `--depth-backend moge2` 启动。

### P2 —— 标定与回归（姿势精度 no-go，后续暂停）

1. ⏸️ 地面掩码内两后端噪声分布及 2 cm 阈值重标定未完成；目前没有受控 DepthPro 同服务平面对照。
2. ❌ qco6j 姿势/网格精度探针 RMSE 超限，P2 精度验收失败；见 §6.5。其余图像姿势回归未继续。
3. ⏸️ 20 次显存增长、独占 GPU 显存余量及 50 次 WSL 稳定性尚未测试。
4. ✅ qco6j 同图五次地面拟合重复性通过；这不替代相对精度回归。

### P3 —— 灰度与默认切换（未启动）

因 §5.3 精度门槛失败，暂停灰度与默认切换。除非后续诊断、校准并重新通过 §6.5 全部精度门槛，不得将 MoGe-2 设为默认或用于生产灰度。MoGe-2 仅作为本轮验证实例运行；验证收尾后应将共享端口 8001 恢复为默认 DepthPro 服务，并以 `/health` 确认后端身份。


---

## 8. 测试与验收清单

| # | 项 | 通过标准 | 当前结果（2026-10-27） |
| --- | --- | --- | --- |
| 1 | 单图 `depth_ms` | MoGe-2 < DepthPro，且达到 §5.3 门槛 | ✅ 单模型基准通过；服务预热后约 87–95 ms（qco6j） |
| 2 | `depth.shape` 对齐 | 三图 × 两后端，全部等于原图 H×W | ✅ 适配层对齐，基准与服务请求成功 |
| 3 | K 语义 | MoGe 的像素 K 与归一化 K 互逆一致（`fx_norm * W == fx_px`） | ✅ adapter 已按宽高分别缩放并使用估计主点 |
| 4 | 地平面稳定性 | 同图重复 5 次，`plane_normal` 夹角 < 0.2°（`random_seed=0` 已固定） | ✅ qco6j 五次输出一致 |
| 5 | 三图精度回归 | 全部满足 §6.5（须 `visualize=true` + `pose_want_mesh=true`，否则没有 `alignment` 报告，见 §6.6） | ❌ qco6j RMSE 超限；另外两张姿势探针未跑 |
| 6 | 回退路径未触发 | `method` 恒为 `camera_ground_anchor`（同上条件） | ✅ qco6j 探针仍为 `camera_ground_anchor`，scale=1.0；但整体精度验收失败 |
| 7 | 显存 | 峰值 < 11 GB（留 1 GB 余量），无 OOM | ⏸️ 未完成独占 GPU 叠加验收；服务探针峰值 4.65 GB（CUDA allocated 口径） |
| 8 | WSL 稳定性 | 连续 50 次请求不崩 VM | ⏸️ 未测试 |
| 9 | 回滚 | `--depth-backend depthpro` 启动后行为与当前版本逐字段一致 | ✅ CLI 回滚后端保留；逐字段服务回归未执行 |

---

## 9. 风险与权衡

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| **深度不是瓶颈** | 换了模型加速有限，却引入标定风险 | P0 先判定，占比 < 25% 直接 no-go |
| **MoGe-2 米制尺度精度未知** | 地平面高度偏移 → 世界系整体漂移 | 三图回归 + `yoga` 探针图 + §6.5 门槛 |
| **主点不再居中** | 点云剪切 → 地面倾角系统性偏差 | 使用 MoGe 自己的 `cx/cy`；回归中单列该指标 |
| **`extract_ground` 2 cm 阈值失配** | 内点过少 → 平面抖动 / 拟合失败 | P2 重标定 + `plane_inlier` 比率门槛 |
| **`spread/0.04` 权重失配** | 只影响可视化网格对齐（边缘功能），不改接口输出的姿势数值（§6.6） | 可延后重标或直接记录为已知偏差；回归中监控 `method` |
| **新依赖 + numpy 版本冲突** | 可能影响服务环境依赖 | 本轮已在独立服务环境安装并验证 MoGe-2 与 `utils3d`；后续升级/复现仍需核对锁定版本 |
| **`moge.model.v2` 是否在已 pin 的 commit 内** | 需升级依赖，可能影响全量流程的 `moge.model.v1` 导入 | 单独验证；v1/v2 模块可共存，但必须实测 |
| **WSL 加载顺序被打乱** | VM 崩溃（历史已知问题） | 适配层保持 `depth → LangSAM → SAM3D` |
| **MoGe-3 范围蔓延** | 1.25 B 参数、Triton 依赖，12 GB 显存吃紧 | 本文明确锁定 MoGe-2 |
| **FP16 数值问题** | 近距离地面深度抖动 | 默认 FP32；FP16 需单独回归通过才允许开 |
| **显存逼近 12 GB 上限 → 慢速模式** | 实测压测期间 `depth_ms` 从 1.85 s 劣化到 3.6–41.7 s，`nvidia-smi` 11.7/12.3 GB；停压数分钟后自愈。会把任何性能对比数据整体带偏 | 测量前独占 GPU（关掉 Windows 侧常驻的 `clip_embedding.py` 等）；每轮采样后 `torch.cuda.empty_cache()`；按 `depth_ms > 5000` 剔除慢样本并对同一现象单独记日志；P2 显存叠加必须在独占环境测 |
| **24 MP 图上换深度收益极低** | 该场景深度只占 12–14%，`extract_ground` 才是 7.5–8.9 s 的大头 | 换模型决策只针对 1024 px 量级输入；大图另立 `extract_ground` 优化项 |

---

## 10. 待确认事项

| # | 事项 | 状态 |
| --- | --- | --- |
| 1 | 是否允许在服务环境安装 `moge` + `utils3d` + `trimesh` | ✅ 已允许并完成安装/验证 |
| 2 | 目标 GPU 是否为当前实测 RTX 4070 12 GB | ✅ 按 RTX 4070 Laptop 12 GB 记录 |
| 3 | 是否接受可能以 no-go 结案 | ✅ 本轮结果为精度门槛失败，默认不切换 |
| 4 | 是否将 DepthPro 非商业许可 → MoGe MIT 作为独立收益 | 未作为抵消精度失败的理由；若需商业决策另行确认 |
| 5 | `yoga_core.jpg` 的 0.3459 m 基线是否仍作为验收探针 | 本轮未执行该姿势/网格探针；后续校准需保留并复测 |
| 6 | 是否把 P0 计时插桩提交进 git | 未提交；等待常规代码审查/提交流程 |
| 7 | 「慢速模式」是否产品化 | 本轮仅测量时人工规避，尚未产品化 |


---

## 11. 本计划明确不做

- 不替换 LangSAM（分割质量与地面 mask 直接耦合，属另一立项）
- 不使用 MoGe-2 的 `normal` 输出做地平面拟合（有新方法风险，且 `extract_ground` 现有 RANSAC 路径已验证）
- 不把网格对齐改成「SAM3D 网格 ↔ 点云」配准（属后续独立事项，触发条件与代价见 §6.6）
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
