# 新增计划：姿势推理接口（复用 ground_service，对齐 ImagePose_Resources 在用接口，输出地面世界系姿势）

> 本版调整：按「**在用接口完整对齐 + 预备移除接口不做兼容**」重写。
> 兼容范围以 `ImagePose_v2/docs/预备移除_旧推理接口与兼容路径_20261021.md` 为准：
> 正式保留的 `GET /v2/pose/health`、`POST /v2/pose/single`、`POST /v2/pose/cancel`、`GET /`
> 与 `retarget_schema=v4` + 内联 bundle、`retarget_schema=v1` 回落 逐项对齐；
> `/v1/*`、`/health`、`/v1/pose/info`、`retarget_schema=v3` 属于预备移除路径，保留现状但不承诺兼容。
> `world`（世界系姿势，含旋转）为本项目增量，P0 核心交付。

## 1. 目标与现状

### 1.1 目标

1. **能力对齐（限在用接口）**：在现有 `ground_service` 进程内提供 `ImagePose_Resources` 仍被 Max 插件使用的接口——`v2` XML 的 health/single/cancel、`GET /`、multipart 上传、`want_overlay`/`want_raw`、`generic_pose`、`cs`（Character Studio）、`target_tpose_bundle` + `retarget_schema` v4、v1 回落、`<targetPose>` XML。`/v1` JSON 系列、`/v1/pose/health`、`/v1/pose/info`、`/health` 别名、`retarget_schema=v3` 属预备移除路径，保留现有实现但不作为对齐目标。
2. **世界空间姿势（本项目增量，P0 核心）**：用 `/v1/ground` 已有的「相机系 → 地面世界系」变换（[rotate_to_up](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L195-L206) + [_world_frame_data](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L209-L252)），把 SAM3D 关节、MHR 关键点、`generic_pose`、重定向骨骼的**位置与旋转**全部换算到地面世界系，一次返回「能直接站到地面 z=0」的姿势。默认开启（`want_world=true`）。
3. **零侵入**：不新增进程、不新增端口、不新增启动脚本，仍以 `python ./ground_pipeline/ground_service.py --port 8004 --host 0.0.0.0` 启动；`/v1/ground`、`/health` 的请求/响应/坐标语义与数值完全不变；SAM3D 权重只加载一次。

### 1.2 现状

- 本服务已有进程内 SAM3D 姿势能力（`/v1/ground` 的 `pose_inference=true`），输出 `pose.joints_3d`（SMPL-24、Figure 系、米，**未叠加 `pred_cam_t`**）、`bbox_2d_xyxy`、`camera`、可选 `overlay`/`mesh`，以及网格与地面配准结果 `pose.alignment`。见 [ground_pose.py](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_pose.py#L205-L265)。
- 不保留 SAM3D 的 `pred_joint_coords`(127×3)、`pred_global_rots`(127×3×3)、`pred_keypoints_3d`(70×3)、`pred_cam_t`、`pred_vertices` 的完整集合（网格路径单独保留），也无任何重定向输出。
- 世界系能力已具备：`/v1/ground` 已输出 `world.ground.rotation` / `world.ground.translation`（[ground_service.py#L1808-L1817](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L1808-L1817)）与 `pose.mesh_world`，与可视化同源。
- 参考服务为独立 Python `http.server` 进程（默认 8010），WSL2 下重定向求解器只能用 `cs_assets/*.py` 源码路径（发布物 `cython/release` 为 Windows cp311 `.pyd`）。

### 1.3 为什么复用同一进程与端口

| 约束 | 说明 |
|---|---|
| 显存 | SAM3D ViT-H 只应加载一份；独立进程会让同卡出现两份权重。 |
| 加载顺序 | 现有 lifespan 已处理「DepthPro 必须先于 LangSAM 初始化 CUDA」的 WSL 约束，新接口不复制这套脆弱逻辑。 |
| 世界系一致性 | 新接口直接复用 `state["segmenter"]`/`state["depth"]` 跑同一份 `extract_ground`，与 `/v1/ground` 的地面平面、世界变换**逐位同源**，避免两套地面估计不一致。 |
| 部署 | 客户端只连 8004；健康检查、日志、systemd 单元不分裂。 |

代价：`/v2/pose/single` 与 `/v1/ground` 共享事件循环与 GPU，长推理互相排队（第 12 节）。

## 2. 与参考服务的能力对齐表

| # | 参考能力 | 本服务落点 | 阶段 | 说明 |
|---|---|---|---|---|
| 1 | `GET /v1/pose/health`（JSON） | 同路径 | **预备移除** | 保留现有实现，不随参考实现演进；新客户端改用 `/v2/pose/health`。 |
| 2 | `GET /health` 作为 v1 health 别名 | **不覆盖** | — | `/health` 保留地面服务语义（`{ok, device}`），避免破坏既有监控。 |
| 3 | `GET /v2/pose/health`（`poseHealth` XML） | 同路径 | **P0 在用** | 复用参考 `health_to_xml`；`status=healthy/disabled`、`api_version="3"`、`gpu` 显卡名、`models{sam3d,vitpose}`，Max 插件握手用。 |
| 4 | `GET /v1/pose/info`（端点目录） | 同路径 | **预备移除** | 保留现有实现；端点目录改由 `GET /` 提供。 |
| 5 | `POST /v1/pose/single`（JSON） | 同路径 | **预备移除** | 保留现有实现，不作为兼容承诺；新客户端只用 `/v2/pose/single`。 |
| 6 | `POST /v2/pose/single`（`poseRequest` XML → `poseResponse` XML，`response_format=json\|xml`） | 同路径 | P0 | v2 默认 XML；`Accept`/`response_format` 可切 JSON。 |
| 7 | `multipart/form-data`（`file`/`image` + 可选 `image_path`） | 同路径 | P0 | 与 `_load_image` 并行支持。 |
| 8 | `image`（data URI / 本地路径 / `file://` / `http(s)`）与 `image_path` | 同 | P0 | 复用 `_load_image`，错误语义共用。 |
| 9 | `request_id` / `engine` | 同 | P0 | `engine` 仅接受 `sam3d_vitpose`；其他值 **200 + `status=error`**（参考实现的业务失败语义）。 |
| 10 | `want_overlay`（默认 **true**，`overlay_jpeg_b64`） | 同 | P0 | 2D 关键点来源见 #20；`world.status=ok` 时先叠加地平面网格。 |
| 11 | `want_raw`（`raw.*` 原始输出） | 同 | P0 | 字段名与坐标系对齐。 |
| 12 | `target_tpose_bundle`（对象/Base64，4 MB 上限，413/400） | 同 | P0 | 指纹服务端始终重算。 |
| 13 | `retarget_schema` = `v4` / `v1` 回落 | 同 | P0 | 在用档位是 `v4`（内联 bundle）与 `v1`（无 bundle 回落，`target_status=skipped`）；`v3` 属预备移除，保留现状。 |
| 14 | `target_status` / `target_fingerprint` / `target_diagnostics` | 同 | P0 | 另增 `target_status=default`（见 6.10）。 |
| 15 | `generic_pose`（`generic_pose_v1`，127 关节/旋转） | 同 | P0 | 默认返回（与参考一致）。 |
| 16 | `cs` / `cs_response_status` / `cs_apply_method` / `matchable` / `bip22_joints` / `bip22_conf` | 同 | P0 | 复用 `cs_assets/*.py`（numpy），不走 cp311 `.pyd`。 |
| 17 | `retarget`（`bip22_pose_v5`）+ `<targetPose schema=bip22_rot_pose_v4>`（v3 属预备移除） | 同 | P0 | JSON 与 XML 双形态。 |
| 18 | `POST /v2/pose/cancel`（`poseCancel` → `poseCancelResult`）；`/v1` 版预备移除 | 同 | P0 | 任务登记 + 阶段间检查；无法中断已在 GPU 上的推理。 |
| 19 | 错误码：业务失败 **200 + `status=error`**，传输/就绪类 400/404/413/500/503 | 同 | P0 | 第 8 节。 |
| 20 | ViTPose 提供 COCO-17 2D（`/v2/pose/single` 内部） | MHR-70 按 `MHR70_FALLBACK` 派生 | P0 降级 | 本机 `models/vitpose-base-simple` 仅有 `config.json`/`preprocessor_config.json`，无权重；字段与语义保持一致，`skeleton_source` 恒为 `MHR127_direct`。权重到位后可切回 ViTPose（P1）。 |
| 21 | 3ds Max 回调脚本 / `cython/release` | 不复制 | — | WSL 环境不可用且与接口能力无关。 |
| 22 | **地面世界系姿势（本项目增量）** | `world` 块 | **P0 核心** | 见 6.9、第 7 节。 |

### 2.1 兼容范围（以预备移除清单为准）

依据 `G:\UGit\QingGeTools\ImagePose_v2\docs\预备移除_旧推理接口与兼容路径_20261021.md`：**在用接口保持兼容，预备移除的旧路径不保证兼容**。

| 类别 | 接口 / 行为 | 本服务处理 |
|---|---|---|
| 在用（已对齐） | `GET /v2/pose/health` | `status=healthy/disabled`、`api_version="3"`、`gpu` 显卡名、`models` |
| 在用（已对齐） | `POST /v2/pose/single`（XML） | `poseRequest` → `poseResponse`；`poseResult`/`csPose`(`applyCharacterStudio`)/`targetPose`(`bip22_pose_v5`)/`restProfile`/`overlay` 复用参考序列化器 |
| 在用（已对齐） | `POST /v2/pose/cancel`（XML） | `poseCancel` → `poseCancelResult`，未知 id 404 |
| 在用（已对齐） | `GET /` | 只读服务说明（端点、输入方式、world 开关） |
| 在用（已对齐） | `retarget_schema=v4` + 内联 `target_tpose_bundle` | `target_status=used`，输出 `targetPose` 与 `profile_rest` |
| 在用（已对齐） | `retarget_schema=v1`（无 bundle 回落） | `target_status=skipped`、`retarget_schema="v1"`，无 `targetPose` |
| 预备移除（不保证） | `POST /v1/pose/single`（JSON）、`POST /v1/pose/cancel` | 保留现有行为，不随参考实现演进 |
| 预备移除（不保证） | `GET /v1/pose/health`、`GET /health`、`GET /v1/pose/info` | `/health` 仍是**地面服务**健康检查，语义与参考不同 |
| 预备移除（不保证） | `retarget_schema=v3` | 仍接受，输出 `bip22_pose_v5`，后续随参考实现一起移除 |

本项目新增（参考没有，不影响旧客户端）：`want_world` / `ground_mode` / `ground` / `want_mesh` 与 `world` / `worldPose`；无 bundle 且未指定 `retarget_schema=v1` 时默认用内建 Biped 重定向（`target_status=default`）。

## 3. 升级范围

### 3.1 P0

- 在 `create_ground_app` 注册 7 条姿势路由 + 1 条说明页：`GET /v2/pose/health`、`GET /v1/pose/health`、`GET /v1/pose/info`、`POST /v2/pose/single`、`POST /v1/pose/single`、`POST /v2/pose/cancel`、`POST /v1/pose/cancel`、`GET /`；沿用 8004。其中 `/v2` 系列与 `GET /` 为在用对齐目标，`/v1` 系列与 `/v1/pose/info` 为预备移除路径（保留但不承诺兼容）。
- 新增 `ground_pipeline/ground_retarget.py`（重定向 + CS + 世界系适配层）：
  - `target_tpose_bundle` 解码/校验/指纹（4 MB 上限）；
  - 请求级 profile 构造与缓存、`retarget_schema` 选择、v4 求解（v3 仅兼容保留）、`bip22_pose_v5` 组装；
  - `cs` 组装（`cs_direct_v6_7`、`bip22_joints`/`bip22_conf`、`points_53`、`mapping_53`）；
  - `world` 换算（位置 + 旋转，第 7 节）。
  - 在 lifespan 中于 SAM3D 之后初始化，结果写入 `state["retarget"]`；初始化失败只让 `/v1|v2/pose/*` 返回 503。
- 扩展 `GroundPoseEstimator`：新增 `include_raw=True` 分支，保留 `pred_joint_coords`/`pred_global_rots`/`pred_keypoints_3d`/`pred_keypoints_2d`/`pred_cam_t`/`global_rot`/`pred_vertices`/`focal_length`；默认关闭，`/v1/ground` 输出不变。
- 直接复用 `ai_vitpose/core/generic_pose.py`、`target_bundle.py`、`target_profile.py`、`target_retarget.py`、`cs_response.py`、`response_serialize.py`、`overlay.py`、`normalize.py`（均无 pyd 依赖）。
- 单人框来源优先级：`bbox_2d` > `person_mask` > LangSAM 人物提示词（就绪时）> 整图框（与参考一致，SAM3D 自身可出结果）。
- `world` 块：默认 `want_world=true`；`ground_mode=auto|request|infer|none`；复用 `extract_ground`（含 `min_vertical_component=0.5` 竖直面剔除）。
- 进程内互斥：`pose_lock`（非阻塞获取，忙时 503），`/v1/ground` 的 `pose_inference` 分支与新接口共用；任务登记表 `state["pose_tasks"]` 支撑 cancel。
- 响应中每个坐标/旋转字段都显式标注 `frame` 与 `unit`，不依赖客户端记忆。

### 3.2 P1

- ViTPose 权重到位后接入 COCO-17 2D（`skeleton_source=coco17_smpl24`），提升 overlay 与 2D 置信度。
- `world.cs`：把 CS 载荷换算到地面世界系。CS 面向 3ds Max 世界（Z 向上、右手），与本项目世界系的「地面法向 = (0,0,-1)」差一个轴向翻转，直接复用会让角色倒置，需显式的 `cs_up_axis` 轴重映射并单独验证。
- 默认目标 profile 的磁盘预构造（参考实现缓存在 `~/.PoseExtractor/server_target_profiles`），缩短首请求耗时。
- 线程池 + 队列化（仍受 `pose_lock` 串行），用于并发请求排队而非阻塞事件循环。

### 3.3 不包含

- 修改 `/v1/ground` 的请求/响应契约、地面算法或既有 `pose` 字段含义；`/health` 不改。
- 新增独立进程、独立端口或独立启动脚本。
- 复制 `cython/release` 的 cp311 `.pyd`、3ds Max 端脚本。
- 下载 ViTPose 权重、复制 SAM3D 权重或参考包中的大体积资产。
- 多人实例分离（一次请求一人）、姿势与地面的物理接触校正、脚底穿透修正、FBX/DCC 导出。

## 4. 服务架构

```text
ground_service（单进程，8004，同一 FastAPI app）
  lifespan：DepthPro → LangSAM → SAM3D → 重定向/CS/世界系适配层
  state = {segmenter, depth, transform, device, pose_estimator, retarget,
           pose_lock, pose_tasks}
  ├─ GET  /health                 既有，地面语义（不改；参考同名接口预备移除）
  ├─ POST /v1/ground              既有，行为与数值不变
  ├─ GET  /                       新增：只读服务说明（在用）
  ├─ GET  /v2/pose/health         新增 XML poseHealth（在用）
  ├─ POST /v2/pose/single         新增 XML/JSON（在用）
  ├─ POST /v2/pose/cancel         新增（在用）
  ├─ GET  /v1/pose/health         旧 JSON（预备移除，保留不兼容）
  ├─ GET  /v1/pose/info           旧端点目录（预备移除，保留不兼容）
  ├─ POST /v1/pose/single         旧 JSON（预备移除，保留不兼容）
  └─ POST /v1/pose/cancel         旧 JSON（预备移除，保留不兼容）
```

就绪与降级矩阵：

| 缺失项 | `/v1/ground` | `/v2/pose/*`（`/v1` 旧路径同源） |
|---|---|---|
| DepthPro / LangSAM | 503（现状） | 可用；`ground_mode=auto` 时 `world.status=skipped`（`error=地面推理未就绪`），`ground_mode=infer` 时 503 `stage=world` |
| SAM3D | 可用（pose 字段报错） | 503 `stage=engine` |
| 重定向资产（`cs_assets`/mapping/profile） | 可用 | 503 `stage=retarget_assets`（`cs` 与 `retarget` 同源资产，不提供半套结果） |

## 5. 目标流程

```text
POST /v2/pose/single（/v1/pose/single 为预备移除的旧路径，逻辑同源）
  ├─ 解析请求（JSON / poseRequest XML / multipart）→ 统一 dict
  ├─ 图像：_load_image / image_path / multipart 字节
  ├─ 单人框：bbox_2d → person_mask → LangSAM 人物提示词 → 整图
  ├─ 目标：target_tpose_bundle 校验 → 请求级 profile；缺省 → 内置默认 Biped profile
  ├─ 任务登记：pose_tasks.begin(request_id)
  ├─ pose_lock（非阻塞；忙 → 503 GPU 忙）
  ├─ SAM3D 推理（state["pose_estimator"]，include_raw=True）
  │    └─ 原始 MHR-127 / 70 / rots / cam_t
  ├─ 归一化：joints_3d(SMPL-24, Figure) / coco17(2D 像素, 3D 相机) / camera
  ├─ generic_pose（figure_v6，米）
  ├─ cs（Figure/Max 语义，位置 cm）
  ├─ retarget（v3 fk_json_folded_gain / v4 chain_ik）→ bip22_pose_v5（figure_v6，米）
  ├─ world（want_world=true）
  │    ├─ 平面来源：请求 ground（plane 或 transform）→ ground_mode=infer 时
  │    │   LangSAM 地面提示词 → DepthPro → extract_ground → rotate_to_up
  │    └─ 位置 p_world = R·(Pᵀ·p_fig + cam_t) + (0,0,-d)
  │      旋转 M_world = (R·Pᵀ)·M_fig·(R·Pᵀ)ᵀ
  ├─ overlay（want_overlay=true）→ 先叠加地平面网格（world.status=ok）→ COCO-17 骨架 → overlay_jpeg_b64
  └─ 输出：v2 XML（poseResponse = poseResult + csPose + targetPose + overlay）；`response_format=json` 时输出同构 JSON
```

关键点：

- 重定向的**输入**必须是 `generic_pose`（Figure 系、米、无 `pred_cam_t` 位移），与参考一致；不得把相机系点云或世界系点云混入。
- 世界系换算只在**输出侧**发生，绝不回写 `generic_pose` / `retarget`，保证与参考逐字段可比。

## 6. 接口契约

### 6.1 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/` | 只读服务说明：`endpoints`、图片输入方式、v2 与 world 开关（**在用**）。 |
| `GET` | `/v2/pose/health` | `poseHealth` XML：`status=healthy/disabled`、`api_version="3"`、`gpu`、`cuda`、`models{sam3d,vitpose}`、`endpoints`、`disabled_reason`（**在用**，Max 客户端握手）。 |
| `POST` | `/v2/pose/single` | 单图推理 + CS + 重定向 + 世界系；`poseRequest` XML 入，`poseResponse` XML 出；`response_format=json` 可出 JSON（**在用**）。 |
| `POST` | `/v2/pose/cancel` | `poseCancel` XML → `poseCancelResult` XML，未知 id 404（**在用**）。 |
| `GET` | `/v1/pose/health` | 旧 JSON 健康检查（**预备移除**，保留但不承诺兼容）。 |
| `GET` | `/v1/pose/info` | 旧端点与输入方式目录（**预备移除**）。 |
| `POST` | `/v1/pose/single` | 旧 JSON 单图推理（**预备移除**）。 |
| `POST` | `/v1/pose/cancel` | 旧 JSON `{request_id}` → `{request_id, cancelled}`（**预备移除**）。 |

### 6.2 请求字段

| 字段 | 类型 | 默认 | 行为 |
|---|---|---:|---|
| `image` | str | — | data URI / 本地路径 / `file://` / `http(s)`；与 `image_path` 二选一。 |
| `image_path` | str | — | 服务端本地路径。 |
| `request_id` | str | `""` | 缺省时服务端生成 UUID；用于日志与 cancel。 |
| `engine` | str | `sam3d_vitpose` | 其他值 **200 + `status=error`**（与参考实现一致）。 |
| `want_overlay` | bool | **true** | 返回 `overlay_jpeg_b64`（JPEG 裸 Base64，无 `data:image` 前缀；失败只写 `overlay_error`）。`world.status=ok` 时先把检测到的地平面网格（世界轴对齐、橙色半透明）投影叠加，再画 COCO-17 骨架；`want_world=false` / 地面 `skipped` 时只画骨架。 |
| `want_raw` | bool | false | 返回 `raw`。 |
| `retarget_schema` | str | `auto` | 缺省 `auto`：有内建默认 profile 时解析为 `v4`（`target_status=default`）；显式 `v1` 则 `target_status=skipped`；`v3` 属预备移除；非法值 400。 |
| `target_tpose_bundle` | str \| object | null | Base64 文本，或 `{text, schema, encoding, fingerprint}`。 |
| `target_fingerprint` | str | `""` | 仅日志/比对，服务端始终重算。 |
| `want_world` | bool | **true** | 返回 `world` 块（本项目增量）。 |
| `ground_mode` | str | `auto` | `auto`：有 `ground` 用之，否则就绪时 `infer`，否则 `skipped`；`request`：缺 `ground` 则 400；`infer`：强制推理平面；`none`：不算世界系。 |
| `ground` | object | null | 二选一：`{plane_normal:[nx,ny,nz], plane_d}`（相机系，`n·p+d=0`）或 `{rotation:3×3, translation:[3]}`（直接传 `/v1/ground` 的 `world.ground.rotation/translation`）。 |
| `want_mesh` | bool | false | 返回 `world.mesh`（相机系网格经世界变换与落地平移，顶点数受 `mesh_vertex_limit` 限制）。网格顶点无论是否返回都会解析，用作落地基准。 |
| `ground_infer_prompts` | object | null | 可选 `{floor_prompt, person_prompt}`，覆盖服务默认提示词。 |

XML `poseRequest`（v2）子元素与 JSON 同名：`image`、`image_path`、`request_id`、`engine`、`response_format`、`want_overlay`、`want_raw`、`target_fingerprint`、`retarget_schema`、`<target_tpose_bundle schema=... encoding=... fingerprint=...>文本</target_tpose_bundle>`；本服务扩展 `want_world`、`ground_mode`、`<ground nx= ny= nz= d= />`。

multipart：`file` 或 `image` 为图片，其余字段为表单文本（布尔接受 `true/false/1/0`）。

### 6.3 JSON 响应字段（v2 `response_format=json` 同构；`/v1` 为预备移除路径）

| 分组 | 字段 | 说明 |
|---|---|---|
| 基础 | `status` | `success` / `error`（与参考一致）。 |
| | `request_id` / `time_ms` / `api_version` | 耗时含推理、CS、重定向与世界系换算。 |
| | `engine` / `skeleton_source` / `orientation_mode` | `sam3d_vitpose` / `MHR127_direct` / `fixed_camera_to_figure_v6`。 |
| 关节 | `joints_3d` | 24×3，Figure 系、米（`frame=figure_v6`，`unit=meter`）。 |
| | `confidence` / `valid_joints` | 24 维置信度与有效数。 |
| 2D/3D 关键点 | `keypoints_2d_coco17` | 17×3（x, y, score）像素。 |
| | `keypoints_3d_coco17` | 17×3，相机系、米。 |
| 相机 | `camera` | `focal_px`、`translation`（`pred_cam_t`）、`image_width`、`image_height`、`calibration`。 |
| 可视化 | `overlay_jpeg_b64` / `overlay_error` | `want_overlay=true` 时；`world.status=ok` 时图中已叠加地平面网格（依赖 `camera.focal_px`，缺失则只画骨架）。 |
| 原始 | `raw` | `want_raw=true` 时，见 6.8。 |
| 通用姿势 | `generic_pose` / `generic_pose_status` / `generic_pose_error` | 见 6.5。 |
| CS | `cs` / `cs_response_status` / `cs_response_error` / `cs_apply_method` / `pose_script_version` / `matchable` / `matchability_reasons` / `bip22_joints` / `bip22_conf` | 见 6.7。 |
| 重定向 | `retarget` / `retarget_schema` / `retarget_status` / `target_status` / `target_fingerprint` / `target_diagnostics` | 见 6.6。 |
| 世界系 | `world` | **本项目核心增量**，见 6.9。 |
| 错误 | `error` | 含阶段前缀：`engine` / `detect` / `generic_pose` / `cs` / `retarget` / `world` / `bundle`。 |

与参考的语义差异（均显式标注）：

1. `skeleton_source` 恒为 `MHR127_direct`（无 ViTPose）；`keypoints_2d_coco17` 由 MHR-70 按 `MHR70_FALLBACK` 派生。
2. `retarget` 存在且 `target_status=default` 时表示使用内置默认目标（参考在此场景为 `not_provided`）。
3. `status=error` 时仍尽量保留 `world=null` 与 `error`，不返回半套 `retarget`。

### 6.4 v2 XML 响应

```xml
<poseResponse api_version="3">
  <poseResult version="1">
    <status>success</status>
    <engine>sam3d_vitpose</engine>
    <request_id>...</request_id>
    <api_version>3</api_version>
    <cs_response_status>ok</cs_response_status>
    <root_missing>false</root_missing>
    <matchable>true</matchable>
    <matchability_reasons/>
    <pose_script_version>cs_direct_v6_7</pose_script_version>
    <cs_apply_method>applyCharacterStudio</cs_apply_method>
    <cs_pose_data_path>cs_pose.xml</cs_pose_data_path>
    <overlay>overlay.jpg</overlay>
    <selfcheck_ok>true</selfcheck_ok>
    <selfcheck_warnings/>
    <missing_53_finger_count>0</missing_53_finger_count>
    <run_log>run_log.txt</run_log>
    <target_status>used</target_status>
    <target_fingerprint>sha256:...</target_fingerprint>
    <retarget_schema>v4</retarget_schema>
    <target_diagnostics>{...}</target_diagnostics>
  </poseResult>
  <csPose version="cs_direct_v6_7" method="applyCharacterStudio">...</csPose>
  <targetPose schema="bip22_rot_pose_v4" method="chain_ik" frame="figure_v6" unit="meter"
              rotation_convention="matrix_3x3_columns_bone_local_axes_in_world">
    <root bone="Bip001 Pelvis" x="0" y="0" z="0"/>
    <bones count="22">
      <bone name="Bip001 Pelvis">
        <global m00="1" ... m22="1"/>
        <local m00="1" ... m22="1"/>
      </bone>
    </bones>
    <diagnostics>...</diagnostics>
  </targetPose>
  <worldPose frame="ground_world_v1" unit="meter" source="inferred">
    <transform rotation="..." translation="..."/>
    <joints count="24">...</joints>
    <bones count="22">...</bones>
  </worldPose>
  <overlay encoding="jpeg_base64">...裸 Base64...</overlay>
</poseResponse>
```

说明：

- `poseResult` 与参考同构；`<targetPose>` 在无 `retarget` 时**不出现**（保持参考的 envelope 形状）。
- `<overlay encoding="jpeg_base64">` 内容与 `overlay_jpeg_b64` 相同：`world.status=ok` 时图中已叠加地平面网格，再叠加 COCO-17 骨架。
- `<worldPose>` 是本项目扩展元素，字段与 6.9 的 JSON 一一对应；`response_format=json` 时以 JSON 返回（`world` 键）。
- `status=error` 时只输出 `poseResult`（含 `<error>`）。

### 6.5 `generic_pose`

与参考一致（`generic_pose_v1`）：`schema_version`、`source=sam3d_vitpose`、`orientation_mode=fixed_camera_to_figure_v6`、`frame=figure_v6`、`pred_joint_coords`(127×3 米)、`pred_global_rots`(127×3×3，`P·R_cam·Pᵀ`)、`pred_keypoints_3d`(70×3 米)、`global_rot`(3，Euler ZYX)、`global_rot_space=zyx_figure_v6`、`pred_cam_t`(3)、可选 `scale_params`/`shape_params`/`bbox`。

默认返回；组装失败时 `generic_pose_status=error`，且**若请求了重定向则整体 500**（不返回未重定向的成功响应）。

### 6.6 `retarget`（`bip22_pose_v5`）与 `<targetPose>`

```json
{
  "schema_version": "bip22_pose_v5",
  "method": "chain_ik",
  "frame": "figure_v6",
  "unit": "meter",
  "rotation_convention": "matrix_3x3_columns_bone_local_axes_in_world",
  "source_profile": "mhr127_figure_v6",
  "target_profile": "mhr_biped_standard",
  "mapping": "mhr_to_mhr_biped_fk_folded_v3",
  "mapping_schema": "mhr_to_biped_mapping_v3",
  "root": {"bone": "Bip001 Pelvis", "source_joint": "root", "position": [0, 0, 0]},
  "bones": {
    "<boneName>": {
      "source_joint": "<mhr joint>",
      "global_rotation": [[1,0,0],[0,1,0],[0,0,1]],
      "local_rotation": [[1,0,0],[0,1,0],[0,0,1]],
      "position": [0, 0, 0]
    }
  },
  "position_semantic": "head",
  "position_space": "figure_v6_world",
  "position_unit": "meter",
  "diagnostics": {"mapped_count": 22, "det_ok": true, "det_issues": [], "chains": {}}
}
```

约定：`bones` 数量由目标 profile 映射决定（标准 Biped 22）；`position` 为骨骼 head；`root.position == bones[root.bone].position`；`global_rotation`/`local_rotation` 列向量为骨骼局部轴方向；`det_ok=false` 或 `unconverged_chains` 非空只作诊断，不升级为硬失败。

### 6.7 `cs`（Character Studio）

- `cs.pose_payload`：`method=applyCharacterStudio`、`version=cs_direct_v6_7`、`frame=ground_world_v1`、`unit=centimeter`、`data[8]`（槽1 厘米位置；头三轴；足法线；胸向右；躯干索引对；手部三轴；`[center, torso_m, fov_deg, aspect]`；校验文件名）。
- `cs.bip22_joints`（22×3，`ground_world_v1` 米）、`cs.bip22_payload.data[0]`（22×3，`ground_world_v1` 厘米，= `bip22_joints*100`）、`cs.points_53` + TSV、`cs.mapping_53`；方向类字段（头三轴 / 足法线 / 胸向右 / 手部三轴 / `bip22_facing`）只做换基、不带平移。
- 顶层同步 `bip22_joints`、`bip22_conf`、`matchable`、`matchability_reasons`、`root_missing`、`selfcheck*`、`missing_53_finger_count`。
- `cs_response_status` 独立于 `status`：CS 失败只写 `cs_response_status=error` 与 `cs_response_error`，不影响 `retarget` 与 `world`。

### 6.8 `raw`

`want_raw=true` 时返回：`bbox`、`mhr70_keypoints_2d`、`mhr70_keypoints_3d`（相机系/米）、`mhr_joint_coords`（相对、相机系/米）、`mhr_joint_global_rots`（相机系）、`pred_joint_coords`、`pred_global_rots`、`pred_keypoints_3d`、`global_rot`、`pred_cam_t`、`scale_params`、`shape_params`。字段名与参考完全一致。

### 6.9 `world`（地面世界系姿势，P0 核心）

```json
"world": {
  "status": "ok",
  "frame": "ground_world_v1",
  "unit": "meter",
  "source": "inferred",
  "ground_mode": "auto",
  "camera_frame": "camera_x_right_y_down_z_forward",
  "rotation": [[1,0,0],[0,1,0],[0,0,1]],
  "translation": [0, 0, -1.114],
  "plane_normal_camera": [0.020, 0.955, 0.296],
  "plane_d_camera": -1.114,
  "ground_normal_world": [0, 0, -1],
  "up_axis_world": [0, 0, 1],
  "camera_height_m": 1.114,
  "camera_position": [0, 0, 1.114],
  "min_z_m": 0.0,
  "ground_snap": {"status": "applied", "source": "sam3d_mesh", "offset_m": -0.012},
  "joints_3d": [[0, 0, 0]],
  "keypoints_3d_coco17": [[0, 0, 0]],
  "bip22_joints": [[0, 0, 0]],
  "generic_pose": {
    "frame": "ground_world_v1",
    "pred_joint_coords": [[0, 0, 0]],
    "pred_keypoints_3d": [[0, 0, 0]],
    "pred_global_rots": [[[1,0,0],[0,1,0],[0,0,1]]],
    "global_rot": [0, 0, 0],
    "global_rot_space": "zyx_ground_world_v1",
    "pred_cam_t_world": [0, 0, 0]
  },
  "retarget": {
    "schema_version": "bip22_pose_v5",
    "frame": "ground_world_v1",
    "position_space": "ground_world_v1",
    "rotation_convention": "matrix_3x3_columns_bone_local_axes_in_world",
    "root": {"bone": "Bip001 Pelvis", "position": [0, 0, 0]},
    "bones": {"Bip001 Pelvis": {"global_rotation": [[1,0,0],[0,1,0],[0,0,1]], "local_rotation": [[1,0,0],[0,1,0],[0,0,1]], "position": [0, 0, 0]}}
  },
  "mesh": {"vertices": [[0, 0, 0]], "faces": [[0, 1, 2]]},
  "error": null
}
```

| 字段 | 说明 |
|---|---|
| `status` | `ok` / `skipped`（`ground_mode=none`、未给平面且无法推理、或模型未就绪）/ `error`（`ground_mode=infer` 且推理失败）。无 `unavailable`。 |
| `source` | `request_plane` / `request_transform` / `inferred`。 |
| `rotation` / `translation` | 相机→世界变换，`p_world = p_cam @ rotation.T + translation`；平面来源时 `translation = [0, 0, -plane_d]`，`request_transform` 来源原样透传，与 `/v1/ground` 的 `world.ground` 完全同源。 |
| `ground_normal_world` / `up_axis_world` | 恒为 `[0,0,-1]` / `[0,0,1]`：世界系右手、**+Z 向上**、地面 `z=0`，`plane_normal`（指向地面）为 -Z。 |
| `camera_height_m` / `camera_position` | `abs(plane_d)` / 平面来源为 `[0,0,-plane_d]`，`request_transform` 来源为透传的 `translation`。 |
| `min_z_m` | 落地后世界系最低点（关节/重定向骨/网格），正常 ≈0；已排除恒为 0 的 `Bip001 root`。 |
| `ground_snap` | 落地结果：`status`（`applied`/`skipped`）、`source`（`sam3d_mesh`=按 SAM-3D 网格最低点 / `joints`=网格缺失时回退关节关键点 / `none`）、`offset_m`（沿世界 +Z 的刚体平移量）。 |
| `joints_3d` / `keypoints_3d_coco17` / `bip22_joints` | 24×3 / 17×3 / 22×3，世界系、米，**已叠加 `pred_cam_t` 与落地平移**；`bip22_joints` 已补回 `cs` 的脚底对齐平移，逐骨与 `joints_3d` 重合（脚趾因取自 MHR-70 有毫米级差异），`bip22_joints[0]` 为落地 root。 |
| `generic_pose` | 与 6.5 同结构，`pred_global_rots` 已换算，`global_rot` 重新按 ZYX 分解。 |
| `retarget` | 与 6.6 同结构，`position_space=ground_world_v1`，位置已叠加 `pred_cam_t` 并含落地平移，旋转已共轭换算。 |
| `mesh` | `want_mesh=true` 时；顶点 `V` 已含落地平移，再 `(V + pred_cam_t)` 经世界变换，`faces` 原样。 |

语义边界：

- `world` 是**新增输出**；`retarget`/`cs`/`generic_pose` 在 `world.status=ok` 时被替换为 `ground_world_v1` 数值，与 `world.*` 逐项一致（`cs` 为同一坐标系的厘米表示），客户端只需按 `frame` 解析一套坐标。`world` 不可用时保持 `figure_v6`，`frame` 标签如实标注。
- **落地作用在 SAM-3D 源数据上**：先按地面算出沿世界 +Z 的位移 `dz`（使 SAM-3D 网格最低点落在 `z=0`，网格缺失时回退到关节/关键点最低点），再把对应的相机系向量 `shift_cam = Rᵀ·(0,0,dz)` 直接加到 `pred_joint_coords`/`mhr_joint_coords`/`pred_keypoints_3d`/`pred_vertices`（`pred_cam_t` 不变），之后 `assemble_pose` 派生的顶层 `joints_3d`、`keypoints_3d_coco17`、`generic_pose`、`cs`、`retarget` 与 `world.*` 全部天然同高，不会出现「姿势一套高度、世界系可视化另一套高度」。`world` 的几何按三步生成：① 由落地后的源数据派生；② 换基到 `ground_world_v1`；③ 叠加 `pred_cam_t`；因 `Q·shift_cam = (0,0,dz)`，等价于世界系直接 +Z 平移 `dz`。落地只做刚体平移，不改肢体相对关系、不改旋转，也**不属于姿势修正**；`rotation`/`translation`/`plane_d_camera`/`camera_height_m`/`camera_position` 是相机-地面测量值，不随落地平移改变，位移量只记录在 `ground_snap.offset_m`。
- `want_world=false`、`ground_mode=none` 或地面不可用（无地面可落）时不做落地，响应保持 SAM-3D 原始高度。
- `want_world=true` 且 `ground_mode=auto` 且 DepthPro/LangSAM 就绪时会额外跑一次地面推理，单请求耗时增加（与 `/v1/ground` 的地面部分同量级），需在响应 `time_ms` 中体现。

### 6.10 默认目标骨架

- 默认目标 = `assets/mhr/mapping/mhr_to_biped_mapping.v3.json` + `assets/biped_rest_profile.v1.json` + `ik_chains.v1.json` 组合 profile（标准 Biped，22 骨、5 条链）。
- 参考服务未提供 bundle 时**不重定向**；本服务为「只发图片即得重定向结果」提供默认目标，`target_status=default`。
- 阶段 0 必须核验：默认 profile 的指纹、必需槽位（`hips`/`head`/四肢/手脚/`spine`/`neck`）、五条 chains 完整性；核验不通过则退回参考语义（`target_status=not_provided`，只返回 `joints_3d`/`generic_pose`/`cs`/`world`，不返回 `retarget`）。

## 7. 坐标系与变换

### 7.1 三个坐标系

| 名称 | 定义 | 手性 | 单位 |
|---|---|---|---|
| 相机系 | x 右、y 下、z 前；地平面 `n·p + d = 0` | 右手 | 米 |
| `figure_v6` | `p_fig = P·p_cam`，`P = [[-1,0,0],[0,0,1],[0,-1,0]]`；+X 右、+Y 后、+Z 上 | **左手**（`det(P) = -1`） | 米 |
| `ground_world_v1` | `R = rotate_to_up(n)`，`p_world = R·p_cam + (0,0,-d)`；地面 `z=0`，+Z 向上，`R·n = (0,0,-1)` | 右手 | 米 |

世界系与 `/v1/ground` 的 `world.ground.rotation/translation` 逐位一致（同一 `rotate_to_up`、同一 `plane_d`），可视化与接口不会出现两套地面。

### 7.2 位置换算

```text
# 落地（先做，作用于 SAM-3D 模型系坐标）
p_cam_src   = p_src + shift_cam,  shift_cam = Rᵀ·(0, 0, dz),  dz = -min_z_world(p_src)

# 再派生 + 换基（p_fig_rootrel 由落地后的源数据生成）
p_fig_rootrel → p_cam_abs = Pᵀ·p_fig_rootrel + pred_cam_t
              → p_world   = R·p_cam_abs + (0, 0, -d)          # 换基 + pred_cam_t 平移
```

- `pred_cam_t` **必须**叠加：SAM3D 的 `pred_joint_coords`/`pred_keypoints_3d`/重定向骨骼位置都是相对量（参考实现同样单独给出 `pred_cam_t`），不叠加会让角色停在相机原点，离地面 `plane_d` 米。
- 顶层 `joints_3d`/`keypoints_3d_coco17`/`bip22_joints`/`retarget`/`generic_pose`/`cs` 在 `world.status=ok` 时由 `apply_world_frames` 统一替换为世界系数值（`cs` 为厘米），不再保留 Figure 系副本；`world` 不可用时两者一致地停留在 `figure_v6`，通过 `*_frame` / `frame` 区分。
- `cs.bip22_joints` 在组装阶段已把脚底对齐到 `z=0`（`bip22_assembler` L5），原点与 `joints_3d` 不同。世界系换算先用 `bip22_source` 中仍标为 `SMPL[n]` 的骨反算该平移（取中位数），补回后再套 `R·Pᵀ`，否则整体会下沉约 0.6 m（实测）；`Bip001 root` 是骨盆水平投影（L4 早于 L5），世界系直接落到 `z=0`。换算结果写入顶层 `bip22_joints` 与 `cs`（厘米），Figure 系原值只在 `world` 不可用时返回。
- 网格：`p_cam = V + pred_cam_t`（`V = pred_vertices`，已含落地），再套世界变换；`faces` 不变。网格顶点**无论 `want_mesh` 与否都会解析**——它同时是落地的基准。
- 落地平移：`dz = -min_z_world`，优先级为 SAM-3D 网格最低点（`source=sam3d_mesh`）→ 关节/关键点最低点（`source=joints`）→ 无可用基准（`status=skipped`，`dz=0`）。命中后直接把 `shift_cam` 加到 `pred_joint_coords`/`mhr_joint_coords`/`pred_keypoints_3d`/`pred_vertices` 上，`pred_cam_t` 保持不变；这只是一次刚体平移，只沿世界 +Z，不改任何旋转与相对姿态，不属于姿势修正。由于换基矩阵 `Q = R·Pᵀ` 满足 `Q·shift_cam = (0,0,dz)`，世界系结果与「换基 + `pred_cam_t` 平移后再 +Z 平移 `dz`」完全等价。

### 7.3 旋转换算

`rotation_convention` 恒为「3×3，列 = 骨骼局部轴在**该 frame** 中的方向」。换基公式：

```text
M_cam   = Pᵀ · M_fig · P                      # figure → camera（参考 P·M_cam·Pᵀ 的逆）
M_world = R · M_cam · Rᵀ                      # camera → world
合并：   M_world = (R·Pᵀ) · M_fig · (R·Pᵀ)ᵀ
```

- `R·Pᵀ` 是正交但 `det = -1`，用**共轭**（左乘 + 右乘转置）而不是单侧相乘：共轭保持 `det(M_world)=det(M_fig)=1` 与正交性；写成 `R·Pᵀ·M_fig` 会得到 `det=-1`，是错误实现。
- `global_rot` 用同一换基后按 ZYX 重新分解，`global_rot_space=zyx_ground_world_v1`。
- `local_rotation` 同样随父链换基：世界系变体中按 `Q·local·Qᵀ`（`Q = R·Pᵀ`）共轭计算，**数值会变化**；仅 `frame` 与 `position_space` 标注为 `ground_world_v1`。`local_rotation` 是相对父骨的旋转，其列方向仍绑定在所在 frame 的轴上，因此换基时必须与 `global_rotation` 用同一共轭公式；实现见 [ground_retarget.py](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_retarget.py#L498-L530)。

### 7.4 单元测试（阶段 0，必须先绿）

1. **往返**：随机 `M_fig`（正交、det=1）→ `M_world` → 反向换基，误差 < 1e-9；全程 `det≈1`、`MᵀM≈I`。
2. **轴向语义**：Figure 的「上」`(0,0,1)` → 相机 `(0,-1,0)`；取法向 `n=(0,1,0)`、`d=-1.1` 的地面，`R·(0,-1,0) ≈ (0,0,1)`，与 `up_axis_world` 一致。
3. **位置**：地面点（`n·p=-d`）变换后 `z≈0`；相机原点变换后 `(0,0,-d)`；`pred_cam_t=(0,0,3)`、`d=-1.1` 在水平地面下得到 `z≈4.1`（相机上方）。
4. **与世界块一致**：`world.rotation/translation` 与 `/v1/ground` 的 `world.ground.rotation/translation` 在同图同参数下逐元素相等。

## 8. 处理与异常规则

| 场景 | HTTP | 响应 |
|---|---:|---|
| JSON/XML 解析失败、字段类型非法、`retarget_schema` 非法 | 400 | `status=error`，指出字段。 |
| `engine` 不支持（非 `sam3d_vitpose`） | **200** | `status=error`，`error="不支持的 engine: ..."`（参考实现的业务失败语义）。 |
| `ground_mode=request` 但 `ground` 缺失/维度错误 | 400 | `status=error`，`stage=world`。 |
| `ground_mode=infer` 但 DepthPro/LangSAM 未就绪 | 503 | `status=error`，`stage=world`；建议客户端改用 `ground` 参数。 |
| 图片超限（沿用 `/v1/ground` 上限） | 413 | `status=error`。 |
| bundle 编码后 > 4 MB 或解码后 > 4 MB | 413 | `status=error`，`target_status=rejected`。 |
| bundle Base64/JSON/schema/unit/coordinate/槽位/chains/非有限值/非单位四元数/hips 名/fingerprint 不一致/`required_slots_ok=false` | 400 | `status=error`，`target_status=rejected`。 |
| `v4` 但 profile 缺 rest profile 或 ik chains | 400 | `status=error`，`target_status=rejected`。 |
| 无可用人物框且 SAM3D 未检出人物 | **200** | `status=error`，`error="detect: 未检出人物（...）"`（参考实现的业务失败语义）。 |
| SAM3D 未配置/加载失败、重定向资产缺失 | 503 | `status=error`，`stage=engine` / `stage=retarget_assets`。 |
| 并发请求占用 `pose_lock` | 503 | `status=error`，`error="GPU 忙，请稍后重试"`（与参考 `GpuBusy` 一致）。 |
| `generic_pose` 组装失败且请求了重定向 | 500 | `status=error`，`stage=generic_pose`。 |
| CS 组装失败 | 200 | 主结果正常，`cs_response_status=error`、`cs_response_error`。 |
| 重定向求解失败 | 500 | `status=error`，`stage=retarget`，`target_status=rejected`。 |
| `ground_mode=infer` 但 `extract_ground` 抛 `ValueError`（无平面/全竖直面） | 200 | `world.status=error`、`world.error`；姿势结果正常返回。 |
| `want_world=true` 但无平面且 `ground_mode=auto` | 200 | `world.status=skipped`、`world.source=null`。 |
| `want_overlay=true` 但绘图失败 | 200 | `overlay_error`。 |
| `cancel` 未知 `request_id` | 404 | `{request_id, cancelled:false, error}`。 |
| `cancel` 命中但推理已在 GPU 上 | 200 | `cancelled=true`；结果在下一检查点返回 `status=error, error="任务已取消"`。 |

补充：

1. 路由用 `async def`（与现有 `/v1/ground` 一致）；内部同步推理，天然串行，仍用 `pose_lock` 保护未来线程池化后的并发。
2. `pose_tasks` 用 `threading.Event`，检查点：SAM3D 之后、`generic_pose` 之后、`retarget` 之前、`world` 之前；**无法中断在途 CUDA 内核**。
3. 日志记 `request_id`、`target_status`、`target_fingerprint`、`retarget_schema`、`mapped_count`、`det_ok`、`unconverged_chains`、`world.source`、`world.status`、`time_ms`；不记图片内容与 bundle 原文。
4. **业务失败一律 HTTP 200 + `status=error`**（含 `engine` 非法、未检出人物），调用方先判 `status` 再看状态码；v1 JSON 与 v2 XML 同语义。
5. 任何 `/v1|v2/pose/*` 异常都不允许影响 `/v1/ground`：适配层失败只置 `state["retarget"]=None`。

## 9. 配置、就绪与资源

- **资源根**：沿用 `SAM3D_RESOURCES=/mnt/g/UGit/QingGeTools/ImagePose_Resources`；启动时设置 `POSEEXTRACTOR_RESOURCES` 与 `POSEEXTRACTOR_CS_ASSETS`，保证 `cs_assets/*.py`、`assets/mhr/**`、`assets/biped_*.json` 可导入/可读。
- **离线**：`HF_HOME=/home/art/.cache/huggingface`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`；不新增下载。
- **求解器**：只用 `cs_assets` 的 `.py`（numpy）；启动自检验证模块可导入并能产出 `bip22_pose_v5`。
- **profile 缓存**：`~/.PoseExtractor/server_target_profiles`；默认目标 profile 在 lifespan 预构造并缓存指纹。
- **显存**：SAM3D 单份；重定向/CS 为 CPU；`world` 不新增模型。启动顺序 DepthPro → LangSAM → SAM3D → 适配层。
- **请求体**：bundle Base64 可达数 MB，需确认反代 body 上限，文档写明 4 MB 与 413 语义。
- **启动方式**：`python ./ground_pipeline/ground_service.py --port 8004 --host 0.0.0.0`（勿用 `-m`；默认端口 8001）。
- **world 依赖**：`ground_mode=infer` 复用 `state["segmenter"]`（`--floor-prompt`/`--person-prompt`）与 `state["depth"]`/`state["transform"]`；缺一即 `world.status=skipped`（`error="地面推理未就绪"`），`ground_mode=infer` 时改抛 503。

## 10. 实施阶段与交付

| 阶段 | 工作 | 完成条件 |
|---|---|---|
| 0. 契约与数学核验 | 核对 `generic_pose`/`bip22_pose_v5`/`targetPose`/`cs` 字段；默认 profile 指纹与槽位；7.4 的四个单元测试 | 字段表与核验脚本可复现；换基单测全绿（含 det/正交/往返） |
| 1. 推理输出扩展 | `GroundPoseEstimator(include_raw=True)` 保留 127/70/rots/cam_t/global_rot/vertices/focal | `/v1/ground` 三张回归图数值不变；新分支数值有限 |
| 2. 输出对齐层 | 归一化（coco17/skeleton24/camera）、`generic_pose`、`cs`、`overlay`、`raw` | v1 JSON 字段与参考逐项对照通过 |
| 3. 重定向适配 | `ground_retarget.py`：bundle 校验、profile、v4 求解（v3 兼容保留）、`bip22_pose_v5`、诊断 | v4 成功；v3 仅回归；非法/超限/缺链错误码正确 |
| 4. 世界系 | `world` 位置 + 旋转 + `ground_mode` 三来源 + `mesh` + 落地平移（`ground_snap`）+ `min_z_m` | 单测通过；`world.rotation/translation` 与 `/v1/ground` 同源；落地后 `min_z_m≈0` |
| 5. 路由与 XML | 7 条姿势路由 + `GET /`、`poseResponse`/`poseHealth`/`poseCancelResult` 序列化、multipart | 同端口 8004 同时跑通 `/v1/ground` 与在用 `/v2/pose/*`、`GET /`；无二次模型加载日志 |
| 6. 文档与示例 | `GROUND_SERVICE_API.md` 新增章节（既有不改）；curl/XML/multipart 示例 | 示例可直接跑通；坐标、单位、错误码与实现一致 |
| 7. 验收 | 第 11 节全项 + 耗时/显存记录 | 全部通过并留档 |

## 11. 验收要点

**能力对齐**

- `GET /v2/pose/health` 返回 `poseHealth` XML：`api_version="3"`、`status=healthy`（就绪）/`disabled`，含 `engine`、`gpu` 显卡名、`cuda`、`models{sam3d,vitpose}`、`endpoints`、`disabled_reason`；Max 插件握手可通。
- `GET /` 返回只读服务说明（`endpoints`、图片输入方式、v2 与 world 开关）。
- `/v1/pose/health`、`/v1/pose/info` 仅保留旧行为，不纳入本轮验收。
- `POST /v1/pose/single`（预备移除，仅回归）返回 `status=success`、`joints_3d`(24×3)、`confidence`(24)、`keypoints_2d_coco17`(17×3)、`keypoints_3d_coco17`(17×3)、`camera`、`overlay_jpeg_b64`、`time_ms`、`cs`、`bip22_joints`/`bip22_conf`、`generic_pose`。
- `POST /v2/pose/single` 用 `poseRequest` XML 请求得到 `poseResponse` XML，含 `poseResult`/`csPose`/`targetPose`/`overlay`；`response_format=json` 时返回 JSON。
- multipart 上传与 `image_path` 均可用。
- 带合法 bundle：`target_status=used`、`target_fingerprint` 与重算值一致、`retarget_schema` 与响应一致；同一图 + 同一 bundle 重复请求旋转逐元素一致。
- 不带 bundle：`target_status=default`，`retarget` 存在；若默认 profile 核验失败则 `not_provided` 且无 `retarget`。
- `retarget` 满足 `schema_version=bip22_pose_v5`、`frame=figure_v6`、`unit=meter`、`rotation_convention=matrix_3x3_columns_bone_local_axes_in_world`，`root.position == bones[root.bone].position`，所有旋转 `det≈1`、正交。
- `<targetPose>` 的 `schema` 与 `retarget_schema` 一致（v3→`bip22_rot_pose_v3`，v4→`bip22_rot_pose_v4`），`<root>` 为米，`null="true"` 仅在缺失时出现。
- 错误：非法/超限/缺槽位 bundle → 400/413 且 `target_status=rejected`；未就绪 → 503；重定向失败 → 500；CS 失败仍 200；`engine` 非法与未检出人物 → **200 + `status=error`**（v2 XML 同样输出 `poseResponse > poseResult > status=error`）。

**世界系（核心）**

- `want_world` 默认 true；`ground` 传 `/v1/ground` 的 `world.ground.rotation/translation` 时，`world.rotation/translation` 与之逐元素相等，`source=request_transform`。
- `ground_mode=auto` 且未传 `ground` 时自动推理地面，`world.source=inferred`，`camera_height_m` 与 `/v1/ground` 同图结果相差 < 1e-6。
- 世界系几何自检：地面点 `z≈0`、`ground_normal_world=(0,0,-1)`、`up_axis_world=(0,0,1)`、`camera_position=(0,0,-1·plane_d)`；落地平移后 `min_z_m≈0`，`ground_snap.status=applied` 且 `offset_m` 等于平移前后同一关节的 z 差。
- 落地基准自检：有 `raw.pred_vertices` 时 `ground_snap.source=sam3d_mesh`（网格最低点落在 `z=0`）；无网格时回退 `joints`；两者都不可用时 `status=skipped`、`offset_m=0`。`rotation`/`translation`/`plane_d_camera`/`camera_height_m` 不随落地平移改变。
- 落地一致性自检：顶层 `joints_3d` / `keypoints_3d_coco17` / `generic_pose` / `retarget` 与 `world` 同源——把顶层位置按 `Q`/`pred_cam_t` 换算后应与 `world.*` 逐元素相等（说明落地只做了一次，未在世界系重复平移）。
- 旋转自检：`world.retarget.*.global_rotation` 全部 `det≈1`、正交；`world.generic_pose.pred_global_rots` 维度 127×3×3 且有限；`world.retarget.*.local_rotation` 同 `global_rotation` 一样按 `Q·M·Qᵀ` 共轭换算，数值与 Figure 版不同（仅 `frame`/`position_space` 标注变化）。
- 位置自检：`world.joints_3d` / `world.bip22_joints` 已叠加 `pred_cam_t`（与未叠加版本相差 `P·R·pred_cam_t` 的平移，逐元素验证）。
- `ground_mode=none` → `world.status=skipped` 且不影响其它字段；`ground_mode=request` 缺 `ground` → 400。

**同进程回归**

- 同一 `--port 8004` 进程同时提供 `/v1/ground` 与 `/v1|v2/pose/*`；`/v1/ground` 默认字段与三张已验收测试图数值不变；`/health` 响应不变。
- 启动日志只出现一次 SAM3D 加载；交替请求后显存峰值不高于改动前 + CS/重定向 CPU 开销。
- 未配置 `SAM3D_RESOURCES`：`/v1/ground` 正常、`/v1/pose/single` 503；重定向资产缺失时同样只影响新接口。
- `cancel` 未知 id → 404；命中后结果返回 `任务已取消`。
- 服务停止后 8004 可再次绑定，无残留进程。

## 12. 风险与后续评估

- **请求互相阻塞**：同事件循环、同 GPU 串行；一次 `pose+world(infer)` 会推迟 `/v1/ground`。P0 接受，文档写明预期耗时；需要并发时再引线程池 + 队列（仍串行占用 GPU）。
- **世界系语义误解**：`ground_world_v1` 是「相机系地面锚定」的结果（+Z 向上、地面 z=0），不是任意建模软件的世界；客户端导入时需按 `up_axis_world` 与 `rotation` 再对接到自己的场景。文档必须给出该提示，避免被当作 Max/Blender 世界。
- **CS 与世界系不同源**：`cs` 保持 Max 语义（位置厘米、Figure 轴向），`world.cs` 留 P1；客户端若同时使用两者，需自行做轴向重映射。
- **figure_v6 左手系 + `det(P)=-1`**：任何跨系换算必须用共轭（7.3），单侧相乘会产生 `det=-1` 的错误矩阵；靠 7.4 单测兜底。
- **默认目标可信度**：默认 Biped 与真实骨架的命名/绑定差异只保证标准语义正确；跨骨架必须上传 bundle。
- **v4 未收敛链**：`unconverged_chains` 非空时局部失真；先只透传诊断，不回退 v3（v3 属预备移除），必要时提示客户端上传带 rest profile 的 bundle。
- **2D 关键点降级**：无 ViTPose 时 `keypoints_2d_coco17` 来自 MHR-70 派生，overlay 精度低于参考；权重到位再切回。
- **请求体积**：bundle Base64 + 图片 data URI 可能触发反代限制，需在部署文档写明上限。
- **根位移策略**：`retarget`（Figure）保持参考「不应用根位移」；`world.retarget` 因叠加 `pred_cam_t` 属于「已放置」，客户端不应重复叠加。

## 13. 已确认决策 与 待确认事项

已确认（2026-10-22）：

1. **`want_world` 默认 `true`**。客户端只发图片即得可直接落地的世界系姿势；不需要时显式传 `want_world=false`，即与参考服务耗时相同。
2. **`ground_mode=auto` 且未传 `ground`、模型未就绪时返回 `world.status=skipped`**，不返回 503。姿势、`cs`、`retarget` 照常交付，`world.error` 写明原因，不阻断主流程。
3. **`world.cs` 留 P1**，本期不做。`cs` 保持参考的 Max 语义（厘米、Figure 轴向）原样输出；`world` 块内不含 `cs` 字段。
4. **默认目标骨架采用参考包的标准 Biped**（`mhr_to_biped_mapping.v3.json` + `biped_rest_profile.v1.json` + `ik_chains.v1.json`），`target_status=default`；未传 bundle 也会返回 `retarget`。
5. **本期只做单人**，不做多人实例分离（与 3.3 一致：一次请求只解析一个框、输出一套姿势）。`multipart/form-data` 按参考支持，但语义是「上传一张单图」，不是多人批量。
6. **兼容范围以预备移除清单为准**（2026-10-22 追加）：`ImagePose_v2/docs/预备移除_旧推理接口与兼容路径_20261021.md` 中「正式保留」的 `/v2` 三接口 + `GET /` + v4/v1 档位逐项对齐；列在「预备移除」的 `/v1/*`、`/health` 别名、`/v1/pose/info`、`retarget_schema=v3` 保留现状、不做兼容承诺。据此，`/v2/pose/health` 的 `status` 输出 `healthy`，`engine` 非法与未检出人物统一 `200 + status=error`。

仍待确认：

7. 是否需要在 `/health` 增量加 `pose_ready`/`retarget_ready`（本计划：不改 `/health`，只加姿势 health 接口；`/health` 属参考侧预备移除路径，保持地面语义）。

## 14. 参考资料

- `G:\UGit\QingGeTools\ImagePose_v2\docs\预备移除_旧推理接口与兼容路径_20261021.md`：兼容范围的权威依据（正式保留 vs 预备移除），以及 `api_version=3`、`status=healthy` 验收要求。
- `G:\UGit\QingGeTools\ImagePose_Resources\SAM3D_POSE_API.md`：端点、`target_tpose_bundle`、v3/v4、`<targetPose>`、`cs`、`raw`、错误码与坐标约定。
- 参考实现：`ai_vitpose/app.py`（路由与请求解析）、`ai_vitpose/core/engine.py`（编排与字段）、`generic_pose.py`（Figure 系构造、`_FIGURE_P`）、`normalize.py`（`camera_points_to_figure_frame`、`skeleton24`）、`cs_response.py`、`response_serialize.py`（`poseResponse`/`poseHealth`/`poseCancelResult`）、`target_bundle.py`、`target_profile.py`、`target_retarget.py`、`overlay.py`、`sam3d_adapter.py`（`MHR70_FALLBACK`）。
- 求解器：`cs_assets/bip22_json_retarget.py`、`bip22_chain_ik.py`、`bip22_assembler.py`、`bip22_rot_retarget.py`、`mhr_retarget.py`。
- 资产：`assets/mhr/mapping/mhr_to_biped_mapping.v3.json`、`assets/biped_rest_profile.v1.json`、`assets/biped_target_profile.v1.json`。
- 本项目：[ground_service.py](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py)（`create_ground_app`、lifespan、`/v1/ground`、`_load_image`、`rotate_to_up`、`_world_frame_data`、`_prepare_pose_mesh_for_views`）、[ground_pose.py](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_pose.py)、[pipeline.py](file:///g:/UGit/Phy-SIC/ground_pipeline/pipeline.py#L26-L37)（`extract_ground`）。
- 文档：`ground_pipeline/GROUND_SERVICE_API.md`（`/v1/ground` 契约，只新增章节）、`ground_pipeline/升级计划_SAM3D_Body姿势推理.md`（既有范围）。
