# ground 服务对接文档（Ground-plane API）

地面提取服务：输入一张 RGB 图片，输出地面平面参数（法向量、平面方程、相机高度）与平面/内点 3D 采样，供上下游服务（SLAM、导航、测距、场景重建等）使用。

- 技术栈：FastAPI + LangSAM（GroundingDINO + SAM2.1）做地面语义分割，DepthPro 估计深度，RANSAC 拟合地面平面
- 运行环境：WSL2（Ubuntu 22.04），Python 3.10，虚拟环境 `.venv-ground`，CUDA 推理

---

## 1. 快速启动

```bat
:: Windows 下双击即可，等价于 WSL 内运行：
::   python ground_pipeline/ground_service.py --host 0.0.0.0 --port 8001
run_ground_pipeline.bat
```

常用启动参数（直接改 [run_ground_pipeline.sh](file:///G:/UGit/Phy-SIC/ground_pipeline/run_ground_pipeline.sh) 或命令行传入）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--host` | `0.0.0.0` | 监听地址 |
| `--port` | `8001` | 服务端口 |
| `--device` | `auto` | `auto` / `cuda` / `cpu` |
| `--sam-type` | `sam2.1_hiera_small` | SAM2 模型类型 |
| `--sam-ckpt` | `data/sam2.1_hiera_small_langsam.pt` | SAM2 权重路径 |
| `--floor-prompt` | `floor, wooden floor.` | 默认地面分割提示词 |
| `--sam3d-resources` | 环境变量 `SAM3D_RESOURCES` | 可选 `ImagePose_Resources` 根目录；配置后尝试加载 SAM3D Body，姿势模块加载失败不影响地面服务 |

> 注意：模型加载约需 **2 分钟**（DepthPro + LangSAM）。加载完成前 `/health` 返回 `ok: false`。服务必须等 `/health` 为 `ok: true` 后才可调用 `/v1/ground`。

## 2. 服务地址

| 环境 | 地址 |
|---|---|
| 本机（Windows 访问 WSL） | `http://127.0.0.1:8001` |
| 局域网其他机器 | 需先以管理员运行 [map_ground.bat](file:///G:/UGit/Phy-SIC/ground_pipeline/map_ground.bat)，然后访问 `http://192.168.139.130:8001` |

---

## 3. 接口

### 3.1 健康检查 `GET /health`

```json
// 200 OK
{
  "ok": true,        // false = 模型尚未加载完成或加载失败
  "device": "cuda"
}
```

**上游调用建议**：先轮询 `/health`（每 2~3s），直到 `ok == true` 再发正式请求。

### 3.2 地面提取 `POST /v1/ground`

请求体为 JSON（Content-Type: `application/json`）。

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `image` | string | 是 | - | 三种格式任选：Base64 data URI、HTTP(S) URL、服务端本地文件路径 |
| `floor_prompt` | string | 否 | `floor, wooden floor.` | 地面分割文本提示词（英文效果更佳，如 `floor, wooden floor.` / `ground, concrete ground.`） |
| `person_prompt` | string | 否 | `person.` | 人物分割文本提示词；**传空串则跳过接触区域检测**。多人物场景可写 `person.` 自动取并集 |
| `contact_threshold` | number | 否 | `0.05` | 接触区域判定距离阈值（米）：人物 mask 反投影点与拟合地面平面距离 ≤ 该值视为接触（近似脚底/贴地部位） |
| `distance_threshold` | number | 否 | `0.02` | 平面拟合距离阈值（米），控制哪些点算"属于地面平面" |
| `iterations` | integer | 否 | `256` | RANSAC 迭代次数 |
| `grid_steps` | integer | 否 | `100` | 平面网格分辨率（返回 `grid_steps × grid_steps` 个平面采样点） |
| `return_data` | bool | 否 | `false` | 为 `true` 时额外返回全量像素级数据：`masks`（四张 PNG base64）、`person_points`（人物点云采样）、`depth_b64`（全量深度 .npy base64）。响应体积明显增大，按需开启 |
| `person_points_limit` | integer | 否 | `2000` | `return_data=true` 时人物点云采样上限 |
| `visualize` | bool | 否 | `false` | 为 `true` 时返回可视化图片（见下方“可视化输出”） |
| `pose_inference` | bool | 否 | `false` | 为 `true` 时，用人物 mask 派生单个图像坐标 2D bbox 并在进程内运行 SAM3D Body ViT-H；不运行 ViTPose。启用姿势模块需配置 `SAM3D_RESOURCES` 环境变量或 `--sam3d-resources` 参数；模块未就绪时响应中会说明姿势错误 |
| `pose_want_overlay` | bool | 否 | `false` | 需同时设置 `pose_inference=true` 和 `visualize=true`；在 `visualization.pose_overlay` 中返回 SAM3D 2D 人体骨架叠加图，不改变原有地面 `overlay` |
| `pose_want_mesh` | bool | 否 | `false` | 需同时设置 `pose_inference=true` 和 `visualize=true`；姿势推理成功时，在 `visualization.pose_mesh` 返回独立 SAM3D 网格多视图图，将完整预测网格放入 `visualization.views` 的相机系/世界系双面板，并额外在 `visualization.pose_world` 返回聚焦于地面世界坐标系的点云 + 网格 3D 渲染图。网格位置未与 DepthPro 点云完成米制配准，仅作视觉对照；独立世界图绘制失败只记录 `pose.mesh_error`，不影响地面结果或其他可视化图

平面拟合内置**竖直面剔除**：相机系 y 轴向下，真实地面法向应满足 `|n_y| ≥ 0.5`（`min_vertical_component`，内部默认 0.5，允许约 60° 的俯仰）。分割掩码里的大面积背板/墙面属于竖直面，RANSAC 会跳过它们，SVD 精化后的平面也会再校验一次；若掩码内只剩竖直面，返回 `422 无法拟合水平地面平面`。该判据避免了「背景墙内点数最多、抢走地面」导致的 `camera_height_m` 异常（如 2.85 m）与后续网格配准被带偏。

#### 请求示例

```bash
# 方式一：本地文件路径（调用方与服务共享文件系统时）
curl -X POST http://127.0.0.1:8001/v1/ground \
  -H "Content-Type: application/json" \
  -d '{"image": "/mnt/g/UGit/Phy-SIC/images/man_exercise2.jpg"}'

# 方式二：Base64 data URI（跨机调用推荐）
curl -X POST http://127.0.0.1:8001/v1/ground \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,/9j/4AAQ...", "floor_prompt": "floor, wooden floor."}'

# 方式三：请求可视化图
curl -X POST http://127.0.0.1:8001/v1/ground \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,...", "visualize": true}'

# 方式四：姿势推理 + 2D 姿势叠加图 + SAM3D 3D 网格多视图图
curl -X POST http://127.0.0.1:8001/v1/ground \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,...", "pose_inference": true, "pose_want_overlay": true, "pose_want_mesh": true, "visualize": true}'
```

#### 响应字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `ok` | bool | 恒为 `true`（失败走 HTTP 错误码） |
| `plane_normal` | [x, y, z] | 地面平面单位法向量（相机坐标系，符号已固定，多次运行输出稳定） |
| `plane_d` | number | 平面方程 `n·p + d = 0` 中的 `d` |
| `camera_height_m` | number | 相机到地面的近似高度（米）＝ `abs(plane_d)` |
| `focal_px` | number | DepthPro 估计的相机焦距（像素） |
| `depth_range_m` | [min, max] | 深度图取值区间（米） |
| `image_hw` | [h, w] | 输入图片尺寸（像素） |
| `floor_pixels` | int | LangSAM 分割出的地面像素数 |
| `plane_inliers` | int | RANSAC 拟合出的地面内点像素数 |
| `person_pixels` | int \| null | 人物 mask 像素数；`person_prompt` 为空或未检出人物时为 `null` |
| `person_bbox_3d_m` | [x1,y1,z1,x2,y2,z2] \| null | 人物 3D 点云轴对齐包围框（相机坐标系，米）：`points[person_mask]` 有效点的 x/y/z min/max |
| `person_centroid_m` | [x, y, z] \| null | 人物 3D 点云质心（相机坐标系，米） |
| `contact_pixels` | int \| null | 接触区域像素数：人物 mask 中反投影 3D 点距地面平面 ≤ `contact_threshold` 的像素。未启用人物分割时为 `null` |
| `contact_centroid_m` | [x, y, z] \| null | 接触区域 3D 质心（相机坐标系，米），无接触时 `null` |
| `contact_bbox` | [x1, y1, x2, y2] \| null | 接触区域在图像上的包围框（像素），无接触时 `null` |
| `masks` | object \| null | **默认关闭返回**，仅 `return_data=true`：`{"floor","inlier","person","contact"}`，各为 8-bit 灰度 PNG base64 data URI（0/255，可还原布尔数组）；`person`/`contact` 无对应数据时为 `null` |
| `person_points` | `[[x,y,z], ...]` \| null | **默认关闭返回**，仅 `return_data=true`：人物 mask 3D 点云采样（相机坐标系，米，≤`person_points_limit` 个） |
| `depth_b64` | string \| null | **默认关闭返回**，仅 `return_data=true`：全量深度图 float32 `.npy` 的 base64（`np.load(io.BytesIO(base64.b64decode(s)))` 还原） |
| `plane_points` | `[[x,y,z], ...]` | 地面上网格采样点（`grid_steps²` 个，相机坐标系，米） |
| `inlier_points` | `[[x,y,z], ...]` | 地面内点 3D 坐标采样（最多 2000 个，随机抽样，相机坐标系，米） |
| `camera` | object | **相机系分组**（地面信息 / 相机信息 / 人物 bbox / 接触），见下方"两系分组输出" |
| `world` | object | **世界系分组**（地面信息 / 相机信息 / 人物 bbox / 接触），见下方"两系分组输出" |
| `visualization` | object \| null | 仅 `visualize=true` 时返回，见下方“可视化输出” |
| `pose` | object \| null | 仅 `pose_inference=true` 时返回；独立于地面坐标系，结构见下方“姿势输出” |

> 顶层旧字段（`plane_normal` / `plane_d` / `plane_points` / `inlier_points` / `person_bbox_3d_m` 等）**全部保留**，与 `camera` / `world` 分组并存，向后兼容。新接入方建议直接使用分组字段。

#### 姿势输出（`pose_inference=true`）

姿势推理是同一 ground 进程内可选的 SAM3D Body ViT-H 能力，不新增 HTTP 服务；bbox 来自 LangSAM 人物 mask 的图像像素范围并作边界扩展。P0 每张图只推理一个合并 mask 对应的 bbox，不运行 ViTPose，也不承诺多人分离。`pose_inference=false` 时省略 `pose` 字段。

| `pose` 字段 | 说明 |
|---|---|
| `status` | `success` / `skipped` / `error`；姿势子任务失败不会改变地面结果 |
| `engine` | `sam3d_body_vith` |
| `skeleton_source` | `MHR127_direct`；从 SAM3D MHR-127 joints 选取 SMPL-24 |
| `joints_3d` | `number[24][3]`，Figure 系位置，单位米；不属于 ground camera/world 系 |
| `joints_3d_coordinate_system` / `joints_3d_unit` | 固定为 `figure` / `m` |
| `orientation_mode` | `fixed_camera_to_figure_v6` |
| `bbox_2d_xyxy` | 人物 mask 派生并扩边裁剪后的图像像素坐标 `[x1,y1,x2,y2]` |
| `camera` | SAM3D 相机估计及输入图像尺寸；`calibration` 表示模型默认 FOV，非实测标定 |
| `time_ms` | 请求内姿势推理耗时；姿势模型不可用或 mask 无效时不一定存在 |
| `error` | `skipped` / `error` 时的简要原因 |
| `overlay_error` | 可选；姿势推理成功但生成 2D 姿势叠加图失败时的错误摘要；`joints_3d` 仍有效 |
| `mesh_error` | 可选；姿势推理成功但网格生成或对应可视化绘制失败时的错误摘要；`joints_3d` 和地面结果仍有效，独立 `visualization.pose_mesh`（若已成功绘制）不受 `views` 绘制失败影响 |
| `mesh_world` | 可选；`pose_want_mesh=true` 且 `visualize=true`、网格成功生成时返回，供客户端自行摆放网格。`vertices`（`number[N][3]`，**世界系**，米，保留 4 位小数）、`faces`（`number[M][3]` 顶点索引）、`vertex_count`、`face_count`、`aligned`（是否已完成 DepthPro 配准，`false` 表示仍是原始模型位置）。世界系定义与 `world` 分组完全一致：`v_world = v_cam @ world.ground.rotation.T + world.ground.translation`，因此可直接与 `world.person.points` 同屏渲染；`alignment` 报告的是相机系变换，不要把两者叠加使用。网格量级约 1.8 万顶点 / 3.7 万面时该字段约 1 MB，仅在客户端确需自行渲染时开启 |
| `alignment` | 可选；`pose_want_mesh=true` 且 `visualize=true` 时返回网格配准状态。流程为**相机系地面锚定（首选，刚体、不缩放）**：用 SAM3D 网格最低分位顶点拟合其自带接触面 → 把接触面法向旋到检测到的相机系地面法向（转角上限 10°）→ 沿地面法向平移使最低分位高度归零。接触面倾角超过上限或顶点不足时回退到**加权关键点 Umeyama → 固定尺度 trimmed ICP → 接触面小角度转平**；随后两种路径统一走**地面接触垂直吸附 → 刚体精修（默认仅沿地面法向平移）**。`status=success` 时字段：`method`（`camera_ground_anchor` 或 `weighted_keypoint_umeyama+depthpro`）、`scale`（地面锚定路径恒为 1.0）、`rmse_m`（关键点内点 RMSE）、`rmse_all_m`、`rmse_before_refine_m`（ICP/接触面转平/吸附后、刚体精修前的关键点 RMSE）、`rmse_final_m`（刚体精修后的关键点 RMSE）、`keypoints_used` / `keypoints_total`、`translation_m`、`keypoint_weight`、`keypoint_diagnostics`（残差最大的至多 12 个关键点，含 `index`、`residual_m`、`weight`、`sam3d_cam`、`depth_cam`）、`camera_ground_check`（原始 SAM3D 网格相对**相机系**地面的高度诊断：`plane_normal`、`plane_d_m`、`mesh_height_m`（`min` / `p1` / `p5` / `p50` / `max`，米）、`mesh_low_plane_tilt_deg` 最低 1% 顶点拟合面相对地面的倾角）、`ground_anchor`（相机系地面锚定报告：`status` 为 `applied` / `skipped` / `error`，`method=camera_ground_anchor`、`scale`（恒 1.0）、`contact_points`、`contact_normal_cam`、`tilt_deg`、`max_tilt_deg`、`applied_tilt_deg`、`pivot_cam`、`axis_cam`、`contact_quantile`、`contact_height_before_m` / `contact_height_after_m`、`shift_m`、`shift_limit_m`（默认 0.30 m）、`lowest_before_m` / `lowest_after_m`；跳过时带 `reason`）、`icp`（`mode=translation`、`trim_ratio`、`trimmed_rmse_m`、`correspondence_mean_m`、`rotation_deg`、`translation_m`、`shift_m`、`limited`）、`contact_plane`（`angle_deg` 接触面相对世界水平面的倾角、`applied_deg` 实际施加转角上限 5°、`max_angle_deg`、`points` 参与拟合的网格低点数、`normal_world`、`pivot_world` 旋转枢轴、`limited`；接触点不足或接触面已水平时为 `skipped`）、`ground_snap`（`source`：`contact_footprint` 用检测到的身体-地面接触区域落位，`mesh_low_quantile` 为无接触点时回退到网格最低分位；`offset_m`、`clusters_with_mesh`、`clusters`（每簇 `contact_points`、`mesh_vertices`、`lowest_before_m`、`gap_after_m`）、`contact_points`、`contact_clusters`、`radius_m`、`lowest_vertex_before_m` / `lowest_vertex_after_m`、`sink_tolerance_m`，仅沿世界重力轴 1-DOF 平移，上限 30cm）、`refine`（末段刚体精修：只优化整体平移（约束到地面法向，1-DOF）/可选的微小整体旋转，姿势与尺度冻结。`status` 为 `applied` / `skipped` / `rejected` / `error`；`method`、`iterations`、`delta_rot_deg`（实际整体转角）、`delta_trans_m`（实际生效的地面法向位移，世界系）、`rotation_limit_deg`（转角上限，默认 0° 即不旋转）、`contact_target_m` / `contact_target_source`、`contact_height_before_m` / `contact_height_after_m`、`penetration_ratio_before` / `penetration_ratio_after`、`max_vertex_shift_m`（默认上限 0.15 m）、`losses`（`contact_m2` / `penetration_m2` / `chamfer_m2` / `projection_m2` / `rotation_reg` / `translation_reg` / `total`））。`status=skipped` 时带 `reason`，网格保持原始模型位置 |

姿势图片不放入 `pose` 对象：请求显式启用 `pose_want_overlay` 和/或 `pose_want_mesh` 并同时设置 `visualize=true` 时，成功绘制的图片分别以独立的 `visualization.pose_overlay`、`visualization.pose_mesh` data URI 返回；原有 `visualization.overlay` 语义不变。`pose_mesh` 由 SAM3D 原始 `pred_vertices` 与模型拓扑绘制，显示模型坐标系，不承诺米制尺度。另在 `pose_want_mesh=true` 时，服务将完整网格按 `pred_vertices + pred_cam_t` 放入 `visualization.views` 的相机系面板，并在 `alignment.status=success` 时先做**相机系地面锚定**（SAM3D 网格自带接触面转平 + 沿相机系地面法向落位，尺度恒 1.0，不缩放；接触面倾角超限才回退到关键点 Umeyama——DepthPro 深度在 MHR-70 关键点像素处采样，与 SAM3D 相机系关键点配对，含离群点剔除），再经 ground 相机到世界系的旋转换算放入世界系面板及独立 `visualization.pose_world`。该配准在整体刚体变换（平移、旋转）之外，末段只对网格整体做**刚体精修**：默认仅沿地面法向平移（1-DOF，横向与纵深沿用前面各阶段的配准结果），可选的微小整体旋转由 `max_rotation_deg` 控制、默认关闭；顶点之间的相对几何、关节角度与尺度全部冻结，因此不会改写 SAM3D 预测的姿势。损失由接触/穿透/人体深度 Chamfer/2D 关键点投影/平移与旋转正则联合构成。实测整体旋转会改变角色在图中的朝向、反而恶化 2D 关键点一致性，而地面接触只需要沿重力轴平移即可达成；整体顶点位移受 `max_vertex_shift_m`（默认 0.15 m）约束，超出则整体拒绝并保留原网格，状态见 `alignment.refine`；配准失败时自动退回原始网格位置，地面结果不受影响。

P0 不返回 ViTPose 2D keypoints/confidence，也不把 `joints_3d` 变换到 ground 的相机系或世界系。资源路径通过服务启动环境变量 `SAM3D_RESOURCES` 或 `--sam3d-resources` 指定到 `ImagePose_Resources` 根目录；需有 `vendor/sam-3d-body` 和 SAM3D ViT-H 权重，并使用兼容 CUDA 的运行环境。配置缺失/加载失败只会使姿势模块不可用，不影响地面服务；请求启用时 `pose.status` 为 `error` 或 `skipped`。

#### 两系分组输出（`camera` / `world`）

同一份几何数据按**相机坐标系**与**世界坐标系**分别组织，每个分组都含 `ground`（地面信息）、`camera`（相机信息）、`person`（人物点云 bbox）、`contact`（接触区域）四个子对象；`person` / `contact` 在未启用或未检出时为 `null`。

```json
{
  "camera": {
    "ground": {
      "plane_normal": [0.021, -0.999, 0.017],   // 相机系单位法向量
      "plane_d": -1.3142,                       // 相机系平面方程 n·p + d = 0 的 d
      "camera_height_m": 1.3142,                // ≈ abs(plane_d)
      "plane_points": [[0.12, -1.31, 1.02], "…共 grid_steps² 个"],
      "inlier_points": [[0.08, -1.30, 1.01], "…共 ≤2000 个"]
    },
    "camera": {
      "focal_px": 891.7,                        // 相机焦距（像素，fx=fy）
      "intrinsics": [[891.7, 0.0, 200.0], [0.0, 891.7, 300.0], [0.0, 0.0, 1.0]],
      "depth_range_m": [0.1, 8.7],
      "image_hw": [600, 400],
      "height_m": 1.3142                        // 相机离地高度（米）
    },
    "person": {
      "pixels": 12345,                          // 人物 mask 像素数
      "bbox_3d_m": [x1, y1, z1, x2, y2, z2],    // 相机系轴对齐包围盒（人物点云 x/y/z min/max）
      "centroid_m": [x, y, z],                  // 相机系质心
      "points": [[x, y, z], "…"]                // return_data=true 时：人物点云采样（相机系，≤person_points_limit）
    },
    "contact": {
      "pixels": 1200,
      "centroid_m": [x, y, z],
      "bbox": [x1, y1, x2, y2]                  // 接触区域图像包围框（像素）
    }
  },
  "world": {
    "ground": {
      "plane_normal": [0.0, 0.0, -1.0],         // ≈(0,0,-1)，指向地面下方
      "plane_d": 0.0,                           // 恒为 0：地面即 z=0 平面
      "camera_height_m": 1.3142,
      "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],   // 3x3 相机系 -> 世界系旋转 R（p_world = p_cam @ R.T + translation）
      "translation": [0.0, 0.0, 1.3142],        // 平移分量，等于相机光心世界坐标
      "plane_points": [[x, y, 0.0], "…"],       // 世界系，z≈0
      "inlier_points": [[x, y, 0.0], "…"]
    },
    "camera": {
      "focal_px": 891.7,
      "intrinsics": [[891.7, 0.0, 200.0], [0.0, 891.7, 300.0], [0.0, 0.0, 1.0]],
      "depth_range_m": [0.1, 8.7],
      "image_hw": [600, 400],
      "height_m": 1.3142,
      "position_m": [0.0, 0.0, 1.3142],         // 相机光心在世界系的位置（地面上方）
      "forward": [0.0, 0.0, 1.0]                // 相机光轴方向（世界系单位向量）
    },
    "person": {
      "pixels": 12345,
      "bbox_3d_m": [x1, y1, z1, x2, y2, z2],    // 世界系轴对齐包围盒（对人物点云在世界系重新取 min/max，方向与地面网格一致）
      "centroid_m": [x, y, z],                  // 世界系质心
      "points": [[x, y, z], "…"]                // return_data=true 时：人物点云采样（世界系）
    },
    "contact": {
      "pixels": 1200,
      "centroid_m": [x, y, z]                   // 世界系（接触点贴近地面，z≈0）
    }
  }
}
```

**世界系变换约定**：取 Rodrigues 旋转 `R` 将相机系地面法向量 `plane_normal` 对齐到 `(0,0,-1)`，即 `wpts = cam_pts @ R.T`，再对 `z` 减 `plane_d`。等价写法：`p_world = p_cam @ world.ground.rotation.T + world.ground.translation`（`translation = (0, 0, -plane_d)`）。于是地面平面成为 `z=0`、法向 `(0,0,-1)`（指向地面下方）、**z 轴向上**；相机光心落在 `position_m = (0,0,-plane_d)`（地面上方 `camera_height_m` 处）。人物立于地面时世界系 bbox 的 `z` 范围从脚底 `≈0` 到头顶正值。可视化图（`views` 右面板、`pose_world`）内部的点云与 SAM3D 网格使用**完全相同的 `R` 与 `plane_d` 平移**，服务端 `pose.mesh_world` 也是按该式生成的，因此客户端摆放网格与返回图片必然一致。

#### 成功响应示例

```json
{
  "ok": true,
  "plane_normal": [0.021, -0.999, 0.017],
  "plane_d": -1.3142160410075063,
  "camera_height_m": 1.3142160410075063,
  "focal_px": 891.6751098632812,
  "depth_range_m": [0.1, 8.7],
  "image_hw": [600, 400],
  "floor_pixels": 65980,
  "plane_inliers": 65624,
  "plane_points": [[0.12, -1.31, 1.02], "...（共 10000 个点）"],
  "inlier_points": [[0.08, -1.30, 1.01], "...（共 2000 个点）"],
  "camera": { "ground": { "...": "相机系地面信息" }, "camera": { "...": "相机内参/高度" }, "person": { "...": "相机系人物 bbox" }, "contact": { "...": "相机系接触" } },
  "world": { "ground": { "...": "世界系地面信息（z=0）" }, "camera": { "...": "相机位姿/高度" }, "person": { "...": "世界系人物 bbox" }, "contact": { "...": "世界系接触" } },
  "visualization": {
    "overlay": "data:image/png;base64,iVBORw0KGgo...",
    "views": "data:image/png;base64,iVBORw0KGgo..."
  }
}
```

#### 可视化输出（`visualize: true`）

响应中的 `visualization` 含地面分析的 `overlay` 与 `views`；姿势骨架图和 3D 网格图可分别独立请求。设置 `pose_want_mesh=true` 时，成功绘制后还返回独立的 `pose_world` 世界视角图。只有对应开关、`pose_inference=true`、`visualize=true` 且姿势推理成功时才返回姿势图。

| key | 内容 | 用途 |
|---|---|---|
| `overlay` | 原图叠加半透明色块：绿色 = LangSAM 分割的地面 mask，蓝色 = RANSAC 平面内点，黄色 = 人物 mask，红色 = 接触区域 | 直观检查分割/拟合是否对准地面，及人物与地面接触位置 |
| `views` | **相机视角与世界 3D 视角左右拼合对比图**：包含场景点云、地面与可选 SAM3D 网格。请求 `pose_want_mesh=true` 时网格也显示于两面板中；`pose.alignment.status=success` 时世界面板中的网格已配准到 DepthPro 人物点云（图例标注 `aligned`），否则保持原始模型位置；网格绘制失败记录在 `pose.mesh_error`，不影响地面输出 | 查看相机/世界整体几何 |
| `pose_overlay` | 原始 RGB 图像叠加 SAM3D Body 的 COCO-17 人体骨架，来自模型原生 MHR-70 2D 关键点；这是独立图片，不改动 `overlay` 和 `views` | 检视 SAM3D 预测的图像平面人体姿态；不依赖 ViTPose |
| `pose_mesh` | SAM3D Body 独立 Front/Side/Back 网格图（模型坐标，不承诺米制尺度） | 检视预测网格形状 |
| `pose_world` | 聚焦于 SAM3D 网格/人物点云的独立世界系 3D 图：DepthPro 场景点云、人物点云、SAM3D 网格及 `z=0` 地面基准；front/side/top 三个正交视图。`pose.alignment.status=success` 时标题标注配准尺度、关键点 RMSE、trimmed ICP 残差与地面吸附位移；配准只修正整体位置/尺度，姿势误差仍在 | 直接观察世界视角下 SAM3D 网格与场景/人物点云的空间分布 |

> 若姿势 3D 推理成功但 2D keypoints 无效或姿势图编码失败，不影响姿势 3D 结果；`pose.overlay_error` 会给出绘图失败摘要，`visualization.pose_overlay` 不返回。网格绘制失败同样不影响姿势关节结果，错误摘要写入 `pose.mesh_error`，且不返回 `visualization.pose_mesh`。

> 注意：`visualize` 会增加约 2~4s 与 ~800KB 响应体积（`views` 需渲染两个 3D 场景面板），仅在需要人工确认时开启。姿势 overlay 和 mesh 均独立按需生成，会额外增加绘图时间和响应体积。

#### 坐标系约定

- **相机坐标系**：原点在相机光心，`x` 向右，`y` 向下，`z` 向前（标准反投影 `x=(u-cx)·z/fx`，`y=(v-cy)·z/fy`），单位为**米**。顶层旧字段、`camera` 分组均使用该坐标系。
- **世界坐标系**：由相机系经 `rotate_to_up(plane_normal, up=(0,0,-1))` 旋转 + 平移得到，地面平面为 `z=0`、**`z` 轴向上**、法向量 `≈(0,0,-1)`（指向地面下方）。`world` 分组使用该坐标系，`camera.position_m` 即相机光心（地面上方 `camera_height_m` 处）。
- `plane_normal`（相机系）主分量被固定为正，因此**朝向可能指向地面也可能反向**；世界系 `world.ground.plane_normal` 已固定为 `≈(0,0,-1)`，无需再归一化。
- `plane_points` / `inlier_points` 可直接用于：地面网格建图、相机高度校验、落地导航平面估计等。

#### 错误响应

| HTTP | detail | 触发条件 |
|---|---|---|
| `503` | 模型未就绪... | 模型尚未加载完成（`/health` 为 false 时） |
| `400` | floor_prompt 不能为空 | `floor_prompt` 传入空串 |
| `404` | 未检出地面区域 | LangSAM 未分割出任何地面 mask（换提示词或图片） |
| `422` | 至少需要 3 个有效地面点 / 无法拟合平面 等 | 深度或分割质量不足，RANSAC 拟合失败 |
| `500` | 分割失败 / 深度推理失败: ... | 模型推理异常（详见服务日志） |

错误响应体格式：

```json
{
  "detail": "未检出地面区域"
}
```

### 3.3 姿势推理 `POST /v1|v2/pose/single`

姿势推理**复用同一个 ground 进程**，不与地面接口抢端口：配置 `SAM3D_RESOURCES`（或 `--sam3d-resources`）后，`lifespan` 会在加载 DepthPro/LangSAM 之后额外加载 SAM3D Body ViT-H 与重定向资产（默认端口 8001；本机联调常用 8004）。未配置时地面接口照常工作，只有姿势路由返回 503。

能力对齐 `G:\UGit\QingGeTools\ImagePose_Resources`（SAM3D MHR-127 → SMPL-24 → `generic_pose` → 目标骨架重定向 → CS 载荷），并新增本项目的地面世界系诊断块 `world`（`ground_world_v1`，+Z 向上、地面 `z=0`）与重力对齐 `align_gravity`（把姿势旋到地面 up 方向并落地）。

| 路由 | 内容 |
|---|---|
| `GET /v1/pose/health` | 就绪状态（JSON） |
| `GET /v2/pose/health` | 就绪状态（XML `poseHealth`，Max 客户端握手用） |
| `GET /v1/pose/info` | 与 `/v1/pose/health` 相同，便于旧客户端探测 |
| `GET /` | 只读服务说明（endpoints / 输入方式 / 世界系开关） |
| `POST /v1/pose/single` | 单张图片推理，响应恒为 JSON |
| `POST /v2/pose/single` | 单张图片推理，响应默认 XML，可用 `response_format` / `Accept` 切 JSON |
| `POST /v1/pose/cancel` | 按 `request_id` 取消排队/进行中的推理（JSON） |
| `POST /v2/pose/cancel` | 同上（XML） |

#### 请求字段

请求体支持 **JSON**、**v2 `poseRequest` XML**、**multipart/form-data**（图片文件 + 其余字段同名文本）。XML 根节点必须为 `poseRequest`。

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `image` | string | 是 | - | 三种格式任选：Base64 data URI、HTTP(S) URL、服务端本地文件路径（可用 `image_path` 代替） |
| `request_id` | string | 否 | 自动生成 | 自定义任务号，用于 `/v1|v2/pose/cancel` 与日志关联 |
| `engine` | string | 否 | `sam3d_vitpose` | 仅支持该值；其它值返回 **200 + `status=error`**（与参考实现一致） |
| `want_overlay` | bool | 否 | `true` | 返回 COCO-17 2D 骨架叠加图（`overlay_jpeg_b64`，**裸 Base64**，无 `data:image` 前缀）；`world.status=ok` 时先叠加检测到的地平面网格 |
| `want_raw` | bool | 否 | `false` | 返回 SAM3D 原始输出（127 关节、旋转、关键点、`pred_cam_t` 等），字段与参考实现同名 |
| `want_world` | bool | 否 | **`true`** | 返回地面世界系诊断块 `world`（`ground_world_v1`）；**不改变**顶层姿势数值的坐标系（顶层恒为 `figure_v6`）。地面未就绪时 `world.status=skipped` / `error` |
| `align_gravity` | bool | 否 | 同 `want_world` | 是否用地面方向做重力对齐并按对齐结果落地；`false` 时不旋转、不落地。未传时回落到 `want_world`（旧客户端兼容）。是否真正推理地面由 `run_ground = want_world \|\| align_gravity` 决定 |
| `want_mesh` | bool | 否 | `false` | `world.mesh` 返回世界系网格顶点/面（约 1.8 万顶点，体积大，按需开启） |
| `ground_mode` | string | 否 | `auto` | 地面来源：`auto`=有 `ground` 就用、否则推理；`request`=必须传 `ground`；`infer`=强制推理（未就绪 503）；`none`=不采用地面变换（注意：它本身不阻止地面推理，需同时令 `want_world=false` 且 `align_gravity=false`） |
| `ground` | object | 否 | - | `{"plane_normal": [x,y,z], "plane_d": d}`（相机系 `n·p+d=0`）或 `{"rotation": 3×3, "translation": [x,y,z]}`，可把 `/v1/ground` 的返回值直接透传 |
| `ground_infer_prompts` | object | 否 | 服务默认 | `{"floor_prompt": "floor, wooden floor."}`，仅在需要推理地面时生效 |
| `retarget_schema` | string | 否 | `auto` | `auto`/`v4`/`v3`/`v1`；`v1` 只返回基础骨架，不做目标骨架重定向 |
| `target_tpose_bundle` | object | 否 | 默认 Biped | 自定义目标骨架包：`{"text", "schema", "encoding", "fingerprint"}`；不传时用内建标准 Biped（`target_status=default`） |
| `target_fingerprint` | string | 否 | - | 校验服务端 profile 指纹 |
| `response_format` | string | 否 | `xml`（v2） | `xml` / `json`，仅 v2 生效 |

```bash
# 只发图片：默认 want_world=true 且 align_gravity 跟随；自动推理地面，
# 顶层 figure_v6 姿势已按重力对齐并落地，同时返回 world 诊断块
curl -X POST http://127.0.0.1:8004/v1/pose/single \
  -H "Content-Type: application/json" \
  -d '{"image": "/mnt/g/UGit/Phy-SIC/images/man_exercise2.jpg"}'

# 复用已算好的地面参数（推荐批量场景），并关闭叠加图
curl -X POST http://127.0.0.1:8004/v1/pose/single \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,...", "want_overlay": false,
       "ground_mode": "request", "ground": {"plane_normal": [0.02, 0.955, 0.296], "plane_d": -1.114}}'

# 不检地面：ViTPose 检人框 + SAM3D 直接推理（相机锚定姿势，不换世界系、不落地）
# 注意 want_world 默认 true、align_gravity 缺省跟随；必须三者同时显式关闭
curl -X POST http://127.0.0.1:8004/v1/pose/single \
  -H "Content-Type: application/json" \
  -d '{"image": "/mnt/g/UGit/Phy-SIC/images/man_exercise2.jpg",
       "want_world": false, "align_gravity": false, "ground_mode": "none"}'

# v2 XML 请求（也可用 multipart 上传图片）
curl -X POST http://127.0.0.1:8004/v2/pose/single \
  -H "Content-Type: application/xml" \
  -d '<poseRequest><image>/data/demo.jpg</image><want_world>true</want_world>
       <ground nx="0.02" ny="0.955" nz="0.296" d="-1.114"/></poseRequest>'
```

#### 响应字段（JSON，v1/v2 共用主体）

| 字段 | 类型 | 说明 |
|---|---|---|
| `status` | string | `success` / `cancelled` / `error` |
| `engine` | string | `sam3d_vitpose` |
| `skeleton_source` | string | `MHR127_direct`（MHR-127 → SMPL-24） |
| `joints_3d` | number[24][3] | `figure_v6` 世界朝向位置，米；`align_gravity=true` 且地面有效时已应用重力旋转并按对齐结果落地，否则为未对齐的相机锚定值 |
| `joints_3d_frame` / `joints_3d_unit` | string | 恒为 `figure_v6` / `meter` |
| `gravity_aligned` | bool | 本次响应是否已应用重力对齐（`false` 时数值未经地面旋转/落地） |
| `align_angle_deg` | number \| null | 地面 up 与 Figure +Z 的夹角（诊断值）；地面不可用、方向退化或未启用时为 `null` |
| `gravity_alignment` | object | 对齐报告：`status`（`applied` / `skipped` / `ground_unavailable` / `ground_error` / `invalid_direction` / `error`）、`applied`、`source`（`sam3d_mesh` / `joints` / `ground` / `none`）、`reason`、`error`、`angle_deg`、`dz_m`、`pivot`。`skipped` 时 `reason=disabled`；失败原因同时以 `warnings` 中的 `gravity alignment unavailable: ...` 暴露 |
| `bbox_source` | string | SAM3D 输入 bbox 来源：`mask`（人物分割）/ `vitpose`（地面不可用时 ViTPose 检人框）/ `full_image`（分割失败整图） |
| `confidence` / `valid_joints` | number[24] / int | 关节置信度 |
| `keypoints_2d_coco17` | number[17][3] | 图像像素坐标 + 分数 |
| `keypoints_3d_coco17` | number[17][3] | 3D 关键点，米；**默认是 SAM-3D 模型系**（根相对，轴向与相机一致：x 右、y 下、z 前），与 `joints_3d` 不同系。仅当 `align_gravity=true` 且对齐成功时被换基到 `figure_v6`，并同时写出 `keypoints_3d_coco17_frame="figure_v6"`；未对齐时该 frame 字段不存在，字段保持模型系 |
| `camera` | object | SAM3D 相机估计（内参为模型默认 FOV，非实测标定）；**相机系**，不随骨骼字段换基 |
| `generic_pose` | object | 参考同构的 127 关节载荷（`pred_joint_coords` / `pred_global_rots` / `global_rot` …），与 `joints_3d` 同属 `figure_v6`；重力对齐时位置按骨盆支点旋转、全局旋转左乘 `S` |
| `bip22_joints` / `bip22_conf` / `bip22_source` / `bip22_height_m` / `bip22_facing` | - | CS 的 22 骨位置，`figure_v6` 系、米，`[0]` 为落地 root（`z=0`）；`bip22_facing` 为同系方向向量 |
| `cs` | object | Character Studio 载荷（`pose_payload`、`points_53`、`bip22_payload` …），`figure_v6` 系；**块内单位不统一**：`pose_payload` / `bip22_payload` 的位置行、`points_53`、`points_53_tsv` 为**厘米**；`bip22_joints` / `bip22_facing` / `bip22_height_m` 为**米**，与顶层同名字段逐项一致；`pose_payload.data[6]` 为 `[center(米), torso_length(米), fov(度), aspect]`；方向类字段无单位。块本身不带 `frame` / `unit` 字段（与 CS 序列化器一致），单位按字段区分；`gravity_aligned` / `align_angle_deg` 与顶层同值 |
| `retarget` | object \| null | 目标骨架重定向结果（`schema_version=bip22_pose_v5`，27 骨，`root` + `bones`，每骨 `global_rotation` / `local_rotation` / `position`）；`frame=figure_v6`、`position_space=figure_v6_world`；`gravity_aligned` / `align_angle_deg` 与顶层同值 |
| `retarget_status` / `retarget_schema` / `retarget_diagnostics` | - | 重定向状态、实际 schema（`auto` 实测解析为 `v4`）、映射诊断 |
| `target_status` / `target_fingerprint` / `target_diagnostics` | - | `default`（内建 Biped）/ `provided` / `rejected` / `not_provided` |
| `profile_rest` / `profile_rest_scope` | - | 目标骨架 rest 姿态（`scope=mapped`） |
| `warnings` | string[] | 非致命告警；重力对齐失败/不可用时追加 `gravity alignment unavailable: <原因>`（显式 `align_gravity=false` 不写告警） |
| `world` | object | **地面世界系诊断块**，仅 `want_world=true`；地面可用时为 `status=ok`，否则为 `skipped` / `error` 占位块。不影响顶层姿势数值 |
| `camera_space` | object | **相机锚定回退**：`want_mesh=true` 且 `world` 不可用时返回，见下 |
| `overlay_jpeg_b64` | string | `want_overlay=true` 时的 JPEG **裸 Base64**（失败写 `overlay_error`）；`world.status=ok` 时图中叠加地平面网格，`want_world=false` 或地面 `skipped` 时只有骨架 |
| `request_id` / `time_ms` | string / number | 任务号与本次耗时 |
| `error` | string | 失败原因 |

> 叠加图内容：`world.status=ok` 时先用 `world.rotation` / `world.plane_normal_camera` / `world.plane_d_camera` 与 `camera.focal_px` 把地平面网格投影到图像（橙色细线、55% 混合，网格与世界轴对齐并以世界原点对齐线段），再画 COCO-17 骨架；因此在同一张图里可直接核对骨架是否站在检测到的地面上。网格仅在 `want_world=true`（默认）且地面成功时绘制，`ground_mode=none` 或 `world.status=skipped/error` 时只返回骨架图。

#### 单位与坐标系约定

服务端不输出「块级 `unit` 就代表块内全部字段」的承诺；单位按字段区分，下表为权威口径。

| 字段 | 坐标系 | 单位 |
|---|---|---|
| `joints_3d`、`generic_pose` 位置、`retarget.*.position`、`bip22_joints`、`bip22_height_m` | `figure_v6` | 米 |
| `keypoints_3d_coco17` | 默认 SAM-3D 模型系；`keypoints_3d_coco17_frame=figure_v6` 时为 `figure_v6` | 米 |
| `cs.pose_payload` / `cs.bip22_payload` 的位置行、`cs.points_53`、`cs.points_53_tsv` | `figure_v6` | 厘米 |
| `cs.pose_payload.data[6][0]`（相机中心）、`cs.pose_payload.data[6][1]`（`torso_length`） | `figure_v6` | 米 |
| `<csPose><camera>/<targetPose><camera>` 的 `center` / `torso_length` | `figure_v6`（`center` 为骨盆→相机的相对向量） | 米 |
| `world.*` 位置、`world.ground_snap.offset_m`、`world.min_z_m`、`camera_space.*` 位置 | `ground_world_v1` / `camera_v1` | 米 |
| `camera.*`、`camera_space.pred_cam_t` | 相机系 | 米 |
| 旋转矩阵 / `global_rotation` / `local_rotation`；`bip22_facing`、头轴、足法线、胸向右、手轴等方向向量 | 随所属块 | 无单位（只换基、不平移） |
| `align_angle_deg`、`gravity_alignment.angle_deg`、`fov` | - | 度 |
| `bip22_conf`、`aspect`、`confidence` | - | 0~1 / 无量纲 |
| `keypoints_2d_coco17`、`bbox_2d_xyxy` | 图像 | 像素 |

> `cs` 的厘米语义来自 Character Studio 序列化（`scale_to_max=100`），只适用于位置槽；`cs.bip22_joints` 与顶层同名字段同为米，不要整体乘/除 100。重力对齐时 `cs` 内的米制与厘米制字段用同一个旋转与同一 `dz`（米 / `dz*100` 厘米）处理。

#### 重力对齐与落地（`align_gravity`）

服务端是重力对齐的**唯一执行方**：客户端拿到 `gravity_aligned=true` 的响应后直接使用数值，不要再做换基、逆共轭或倾斜修正。

- **触发**：`run_ground = want_world || align_gravity` 决定是否解析/推理地面；`align_gravity=true` 且地面变换可用时执行对齐。
- **旋转**：把地面 up 方向在 Figure 系中的表示 `u_fig = FIGURE_P @ Rᵀ @ e_z` 旋到 `(0,0,1)`，得到真旋转 `S`（Rodrigues，`det=+1`）；`align_angle_deg = acos(u_fig_z)`。
- **作用范围**：`joints_3d`、`keypoints_3d_coco17`（先由模型系换基到 `figure_v6`，再随其余 `figure_v6` 字段一起绕骨盆支点旋转）、`generic_pose`（位置绕骨盆支点旋转、`pred_global_rots` 左乘 `S`、`global_rot` 欧拉角同步）、`retarget`（每骨 `global_rotation` 左乘 `S`、`position` 绕骨盆旋转）、`cs`（`bip22_joints` 等米制字段绕 `bip22_joints[1]` 支点旋转，`pose_payload` / `points_53` 等厘米字段按同一支点换算，仅旋转不缩放；`pose_payload.data[6][0]` 相机中心是「骨盆→相机」的相对向量，只做 `center' = S·center`（相机随姿势刚性旋转），**不**绕骨盆支点平移、**不**叠加落地 `dz`，`data[6][1]` 的 `torso_length` 保持不变）；`camera` 块恒为相机系，不参与对齐。
- **落地**：对齐后在 mesh 顶点（缺网格时回退 joints / keypoints）上采样最低 `z`，统一取 `dz = -min_z`，所有位置 `z += dz`；`Bip001 root` 恒置 `z=0`。落地只做平移，不改旋转与朝向。
- **幂等**：对同一 payload 重复调用会报 `gravity alignment already applied`，不会二次旋转。
- **元数据**：顶层 `gravity_aligned` / `align_angle_deg` / `gravity_alignment`；`cs`（含 `cs.pose_payload`）与 `retarget` 镜像同名两项。

`gravity_alignment.status` 取值与数值影响：

| status | 含义 | 数值是否改动 |
|---|---|---|
| `applied` | 对齐并落地成功 | 已改动 |
| `skipped`（`reason=disabled`） | `align_gravity=false`；地面变换存在时 `angle_deg` 仍给出诊断角 | 未改动 |
| `ground_unavailable` | `align_gravity=true` 但地面未就绪/未推理 | 未改动 |
| `ground_error` | 地面推理报错 | 未改动 |
| `invalid_direction` | 地面方向退化（`degenerate` / `opposite` / 非正交 / 非真旋转） | 未改动 |
| `error` | 其他对齐异常 | 未改动 |

未对齐时 `gravity_aligned=false`、`align_angle_deg=null`，并在 `warnings` 追加 `gravity alignment unavailable: <原因>`；`source` 取落地采样来源 `sam3d_mesh` / `joints`，未执行时为 `ground`（地面变换可用）或 `none`。

**无地面路径**（`want_world=false` + `align_gravity=false` + `ground_mode=none`）完全不推理地面：bbox 由 ViTPose 检出（`bbox_source=vitpose`），SAM3D 直接输出相机锚定的 `figure_v6` 姿势，`gravity_alignment.status=skipped`。

#### `world` 块（地面世界系，仅诊断）

与 `/v1/ground` 的 `world` 分组**同源**：同一 `rotate_to_up(plane_normal)` 与 `plane_d`，`ground_normal_world≈[0,0,-1]`、`up_axis_world≈[0,0,1]`、地面 `z=0`。

| 字段 | 说明 |
|---|---|
| `status` | `ok` / `skipped`（`ground_mode=none` 或地面推理不可用）/ `error`；非 `ok` 时只有 `error` 与坐标系元数据 |
| `source` | `inferred` / `request_plane` / `request_transform` |
| `rotation` / `translation` | 相机→世界：`p_world = p_cam @ rotation.T + translation` |
| `plane_normal_camera` / `plane_d_camera` / `camera_height_m` / `camera_position` | 相机系平面参数与相机光心世界坐标（测量值，不随落地平移改变） |
| `ground_snap` | 落地结果：`status`（`applied`/`skipped`）、`source`（`sam3d_mesh`=按 SAM-3D 网格最低点 / `joints`=网格缺失时回退关节关键点 / `none`）、`offset_m`（沿世界 +Z 的刚体平移量）；注意与 `/v1/ground` 的 `mesh.alignment.ground_snap`（网格配准接触落位报告）同名但语义不同 |
| `min_z_m` | 落地后人物最低点（关节/重定向骨/网格），正常情况下 ≈0；不含恒为 0 的 `Bip001 root` |
| `joints_3d` | number[24][3]，世界系、米，已叠加 `pred_cam_t` 与落地平移 |
| `keypoints_3d_coco17` | number[17][3]，世界系、米 |
| `bip22_joints` | number[22][3]，世界系、米；已补回 CS 的「脚底对齐」平移，逐骨与 `joints_3d` 重合（脚趾取自 MHR-70，毫米级差异），`[0]` 为落地 root（`z=0`） |
| `generic_pose` / `retarget` | 同源载荷的世界系副本：位置换基，`global_rotation` / `local_rotation` 均按共轭换算 |
| `mesh` | 仅 `want_mesh=true`：`vertices`（世界系、米）+ `faces` |

> **`world` 是诊断块，不参与顶层姿势数值**。其几何按三步生成：① 复制一份 SAM-3D 原始输出，沿世界 +Z 落地——算出 `dz` 使网格最低点落在 `z=0`（网格缺失时回退关节/关键点最低点），把相机系向量 `shift_cam` 加到 `pred_joint_coords` / `mhr_joint_coords` / `pred_keypoints_3d` / `pred_vertices` 上（`pred_cam_t` 不变）；② 用这份已落地副本按 `figure_v6`/相机系规则派生；③ 换基到 `ground_world_v1` 并叠加 `pred_cam_t`。因换基矩阵满足 `Q·shift_cam = (0,0,dz)`，结果与世界系直接 +Z 平移 `dz` 等价。落地只是刚体平移，不改肢体相对关系、不改旋转，也不改 `rotation`/`translation`/`plane_d_camera` 等相机-地面测量值；位移量见 `ground_snap.offset_m`。地面不可用时不生成该副本，`world.status` 为 `skipped` / `error`。

> 顶层 `joints_3d` / `generic_pose` / `retarget` / `bip22_joints` / `cs` **恒为 `figure_v6`**；`keypoints_3d_coco17` 例外——默认是 SAM-3D 模型系，只有重力对齐成功后才换基到 `figure_v6`（并写 `keypoints_3d_coco17_frame`）。它们与 `world` 内的世界系副本不是同一坐标系，不可混用。最终交付姿势请读顶层（`align_gravity=true` 时已对齐并落地）；需要 `ground_world_v1` 数值时读 `world.*`；`camera` 始终是相机系。

#### `camera_space` 块（相机锚定回退）

地面不可信（`want_world=false`，或地面推理失败导致 `world.status!=ok`）且 `want_mesh=true` 时返回。**全程不做地面平面换基、不做落地平移**，只给出 SAM-3D 原始几何在相机系的表示。

| 字段 | 说明 |
|---|---|
| `frame` / `unit` / `convention` | `camera_v1` / `meter` / `x right, y down, z forward` |
| `pred_cam_t` | `[3]`，SAM-3D 模型系 -> 相机系的平移 |
| `vertices` | number[N][3]，相机系、米（`pred_vertices + pred_cam_t`） |
| `faces` | number[M][3]，三角面索引 |
| `keypoints_3d_coco17` | number[17][3]，相机系、米（`pred_keypoints_3d + pred_cam_t`，与网格同系，可直接叠画） |

> 顶点与关键点都来自 SAM-3D 模型系，统一加 `pred_cam_t` 后落在相机系，因此骨架与网格天然重合。本块恒为原始相机锚定几何，与顶层是否已做重力对齐无关；需要相机系可视化时请直接用本块。

#### v2 XML 响应

`POST /v2/pose/single` 默认返回 `application/xml`：`<poseResponse api_version="3">` 内含 `<poseResult>`（状态、关节、重定向骨、`<restProfile>` 等）、`<csPose>`（`figure_v6` 数值、厘米；例外：`<camera torso_length=... fov=... aspect=...>` 与 `<center x y z>` 为米，与 CS 序列化器一致）与 `<targetPose>`（`frame="figure_v6"`），并在根节点末尾（`</poseResponse>` 之前）插入本项目扩展的 `<worldPose>`（`ground_world_v1`，仅 `want_world=true` 且地面可用）。

重力对齐元数据以**属性**形式统一写在 `poseResult` / `csPose` / `targetPose` 三个节点上，不新增子节点：`gravity_aligned="true|false"`；角度有限时附 `align_angle_deg="12.860000"`，为 `null` 时省略该属性。

`<targetPose>` 在 `<root>` 之后、`<bones>` 之前带一个 `<camera torso_length="米" fov="度" aspect="无量纲"><center x y z/></camera>`，字段语义与 `<csPose><camera>` **完全一致**：`center` 是「骨盆→相机」的相对向量（`figure_v6`、米），`torso_length` 为该视角下躯干长度（米）。无相机槽时写 `<camera null="true"/>`。这样只消费 `targetPose` 的通道也能匹配图片视角；`align_gravity=true` 时它随 `<csPose><camera>` 一同按 `center' = S·center` 旋转，不含落地 `dz`。

```xml
<?xml version='1.0' encoding='utf-8'?>
<poseResponse api_version="3">
  <poseResult version="1" gravity_aligned="true" align_angle_deg="12.860000"> ... </poseResult>
  <csPose version="cs_direct_v6_7" method="applyCharacterStudio" gravity_aligned="true" align_angle_deg="12.860000"> ... </csPose>
  <targetPose schema="bip22_pose_v5" method="..." frame="figure_v6" unit="meter" gravity_aligned="true" align_angle_deg="12.860000"> ... </targetPose>
  <restProfile ...> ... </restProfile>
  <worldPose frame="ground_world_v1" unit="meter" source="inferred" status="ok">
    <transform rotation="..." translation="0.000000 0.000000 1.114000"/>
    <joints count="24">
      <joint index="0" x="..." y="..." z="..."/>
    </joints>
    <bones count="27">
      <bone name="Bip001 Pelvis">
        <global matrix="..."/>
        <local matrix="..."/>
        <position x="..." y="..." z="..."/>
      </bone>
    </bones>
  </worldPose>
  <overlay>...</overlay>
</poseResponse>
```

#### 健康检查与取消

```json
// GET /v1/pose/health（/v2/pose/health 为同内容的 poseHealth XML）
{
  "status": "healthy",             // healthy / disabled（Max 客户端兼容 ok）
  "engine": "sam3d_vitpose",
  "api_version": 3,
  "gpu": "NVIDIA GeForce RTX 4090",
  "cuda": true,
  "models": {"sam3d": "sam-3d-body-vith", "vitpose": "disabled"},
  "world": {"ready": true, "segmenter": true, "depth": true},
  "retarget": {"ready": true, "skeleton_id": "mhr_biped_standard", "bone_count": 66, "mapping_bone_count": 27, "fingerprint": "sha256:...", "chains": ["spine", "neck", "finger_L_Finger0", "finger_L_Finger1", "finger_R_Finger0", "finger_R_Finger1"]},
  "endpoints": ["GET /v1/pose/health", "..."],
  "disabled_reason": "",
  "skeleton_source": "MHR127_direct"
}
```

```bash
curl -X POST http://127.0.0.1:8004/v1/pose/cancel -H "Content-Type: application/json" \
  -d '{"request_id": "6f87f78bbb8b44cbb80cf874365e5258"}'
# 200 {"request_id": "...", "cancelled": true}
# 404 {"request_id": "...", "cancelled": false, "error": "未找到任务"}
```

推理本身是同步的：`request_id` 只在请求执行期间登记，取消用于释放排队等待与中途放弃，已返回的 200 响应不会被改写。同一时刻只允许一个姿势任务占用 GPU，并发请求返回 503 `GPU 忙，请稍后重试`。

#### 错误响应

| HTTP | 触发条件 |
|---|---|
| `400` | JSON/XML 解析失败、字段类型非法、`ground_mode` 不支持、`ground_mode=request` 但缺 `ground`、bundle 校验失败（`target_status=rejected`） |
| `404` | `cancel` 的 `request_id` 未找到 |
| `413` | 请求体或图片超过 40 MB |
| `500` | 推理异常、`generic_pose` 组装或重定向失败 |
| `503` | SAM3D 未配置/加载失败、重定向资产缺失、`ground_mode=infer` 但地面模型未就绪、GPU 忙 |

错误响应体为 `{"status": "error", "engine": "sam3d_vitpose", "error": "stage: ...", "request_id": "...", "target_status": "..."}`（v2 XML 同样生效，错误也按 XML 输出）。

> 业务失败沿用参考实现语义：**HTTP 200 + `status=error`**，包括 `engine` 非法与 SAM3D 未检出人物。调用方应先判断 `status`，不要只看状态码。

#### 与参考 ImagePose_Resources 的兼容范围

依据 `ImagePose_v2/docs/预备移除_旧推理接口与兼容路径_20261021.md`：**在用接口保持兼容，预备移除的旧接口不保证兼容**。

| 类别 | 接口 / 行为 | 说明 |
|---|---|---|
| 在用（已对齐） | `GET /v2/pose/health` | `status=healthy/disabled`、`api_version=3` 属性、`gpu` 显卡名、`models` 字段 |
| 在用（已对齐） | `POST /v2/pose/single`（XML） | `poseRequest` → `poseResponse`；`poseResult` / `csPose`(`applyCharacterStudio`) / `targetPose`(`bip22_pose_v5`) / `restProfile` / `overlay` 均复用参考序列化器 |
| 在用（已对齐） | `POST /v2/pose/cancel`（XML） | `poseCancel` → `poseCancelResult`，未找到 404 |
| 在用（已对齐） | `GET /` | 只读服务说明 |
| 在用（已对齐） | `retarget_schema=v4` + 内联 `target_tpose_bundle` | `target_status=used`，输出 `targetPose` 与 `profile_rest` |
| 在用（已对齐） | `retarget_schema=v1`（无 bundle 回落） | `target_status=skipped`、`retarget_schema=v1`，无 `targetPose` |
| 预备移除（不保证） | `POST /v1/pose/single`（JSON）、`POST /v1/pose/cancel` | 保留旧行为，不随参考实现演进 |
| 预备移除（不保证） | `GET /v1/pose/health`、`GET /health`、`GET /v1/pose/info` | `/health` 在本服务中仍是**地面服务**健康检查，语义与参考不同 |
| 预备移除（不保证） | `retarget_schema=v3` | 仍接受，输出 `bip22_pose_v5`，后续随参考实现一起移除 |

本项目新增（参考实现没有，不影响旧客户端）：`want_world` / `align_gravity` / `ground_mode` / `ground` / `want_mesh`，以及 `gravity_aligned` / `align_angle_deg` / `gravity_alignment` / `bbox_source` 元数据与 `world` / `worldPose` 块；无 bundle 且未指定 `retarget_schema=v1` 时，默认用内建 Biped 重定向（`target_status=default`）。

---

## 4. Python 调用示例

```python
import json
import base64
import urllib.request

BASE = "http://127.0.0.1:8001"

def wait_ready(timeout_s=240):
    """轮询 /health 直到服务就绪。"""
    import time
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=2) as r:
                if json.loads(r.read())["ok"]:
                    return True
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError("ground service not ready")

def ground(image_path: str, prompt: str = "floor, wooden floor.", visualize: bool = False):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    req = {
        "image": f"data:image/jpeg;base64,{b64}",
        "floor_prompt": prompt,
        "visualize": visualize,
    }
    body = json.dumps(req).encode()
    request = urllib.request.Request(
        BASE + "/v1/ground", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as resp:
        return json.loads(resp.read().decode())

wait_ready()
result = ground("/path/to/photo.jpg", visualize=True)
print(result["camera_height_m"], result["plane_normal"], len(result["plane_points"]))

# 保存可视化图（data URI -> 文件）
import re
for key in ("overlay", "views"):
    uri = result["visualization"][key]
    raw = re.sub("^data:image/png;base64,", "", uri)
    with open(f"/tmp/ground_{key}.png", "wb") as f:
        f.write(base64.b64decode(raw))
```

---

## 5. 与 embedding 服务（qwen3-vl）联用

两服务可同时运行（已验证，合计显存约 7.8GB，12GB 显卡留约 4GB 余量）：

| 服务 | 端口 | 启动入口 | 健康检查 |
|---|---|---|---|
| embedding（图文/文本向量） | `8000` | `run_embedding.bat` | `GET http://127.0.0.1:8000/health` |
| ground（地面提取） | `8001` | `run_ground_pipeline.bat` | `GET http://127.0.0.1:8001/health` |

典型串联：embedding 服务判断画面语义/检索参考图 → 命中后调 `/v1/ground` 提取地面平面参数用于几何任务。

---

## 6. 对接注意事项

1. **冷启动慢**：模型加载约 2 分钟，务必先轮询 `/health`。
2. **单次请求耗时**：`/v1/ground` 一次调用约 10~30s（大图更慢，示例 600×400 图片约 15s），请设置 ≥120s 的超时。
3. **超长图片**：DepthPro 输入会按比例缩放，SAM2 对超大图（如 4000+ 像素）耗时会明显上升；建议上游先做长边 ≤2048 的缩放。
4. **不要动模型加载顺序**：`lifespan` 中必须先加载 DepthPro 再加载 LangSAM（SAM2 若作为第一个 CUDA 模型会导致 WSL VM 崩溃），请勿调整该顺序。
5. **本机免端口映射**，仅局域网跨机访问需要 [map_ground.bat](file:///G:/UGit/Phy-SIC/ground_pipeline/map_ground.bat)。
6. **OpenAPI 文档**：服务自带 Swagger，浏览器打开 `http://127.0.0.1:8001/docs` 可在线调试。
