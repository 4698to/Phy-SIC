# ground_pipeline 升级计划：集成 SAM3DPose 姿势推理模块

## 1. 目标与现状

升级的是 `ground_pipeline` 自身的处理流水线：在现有地面分析步骤中集成 SAM3DPose 姿势推理模块，使一次图片处理可同时产出地面几何、人物-地面接触信息和人体关节姿势。不是新增一个对外姿势服务，也不改变既有 `/v1/ground` 调用行为；姿势推理作为可选流水线能力，关闭时跳过该模块。

当前流水线为：输入图片 → LangSAM 地面分割 → DepthPro 米制深度 → 反投影和 RANSAC 地面拟合 → 可选人物 mask、人物 3D 包围框与地面接触区 → 输出相机系/世界系结果。人物分析依赖 LangSAM 的 `person_prompt`，目前没有骨架或关节姿势输出。

流程与 SAM3DPose 实现细节参考 `G:\UGit\QingGeTools\ImagePose_Resources`：第一阶段直接使用现有 `person_mask` 在图像像素平面计算单人 2D 外接框，再将 RGB 图像和 bbox 输入 SAM 3D Body ViT-H 做 Body 姿势推理；第一阶段不集成、不运行 ViTPose。参考资源当前将 SAM3D 适配器与 ViTPose 分开封装在 `ai_vitpose/core`，使用本地模型资源；姿势输出包含关节位置、置信度、关键点及可选 overlay/raw 数据。本计划只复用/集成 SAM3DPose 推理所需模块，不新增对外 HTTP 服务。

## 2. 升级范围

### P0：流水线内姿势推理闭环

- 将 SAM3DPose 推理能力集成到 `ground_pipeline` 的服务启动和单图处理流程中；第一阶段仅加载并复用 SAM3D Body 推理模型，不加载 ViTPose。
- 姿势输入框由现有 `person_mask` 的图像像素坐标计算为单个 2D bbox（附带与参考实现相当的适度边界扩展）；不使用 3D 点云 bbox 作为模型输入。
- 为 `/v1/ground` 增加可选姿势开关；关闭时不运行姿势推理，保持现有处理和响应兼容。
- 开启时复用已解码 RGB 图像和人物 mask 运行姿势模块，与现有地面/人物链路共享同一输入图像。
- 将姿势结果作为独立 `pose` 对象加入地面响应；姿势推理失败不使地面提取失败。
- 补充模块资源路径、CUDA/依赖配置、模型就绪状态与错误日志说明；不新增姿势 HTTP API。

### P1：体验与质量完善

- 在现有可视化输出中可选叠加 SAM3DPose 2D 姿势 overlay，且不改变现有地面 overlay 的定义。
- 可独立请求 SAM3D Body 预测网格多视图图 `visualization.pose_mesh`；请求网格时将完整预测网格按 `pred_vertices + pred_cam_t` 放入 `visualization.views` 的相机系面板并转换到世界系面板，同时增加 `visualization.pose_world`，提供聚焦于网格/人物点云的独立世界视角渲染图。网格位置未与 DepthPro 点云完成米制联合标定，不代表真实空间对齐或米制人体重建；网格生成/绘制失败通过 `pose.mesh_error` 报告，不影响关节、地面结果或已经生成的其他可视化图。
- 第一阶段只覆盖人物提示得到的单个姿势结果；若 mask 中包含多个人，只按当前整体 mask 形成一个 bbox，不承诺多人分离或实例级匹配。待第一阶段完成后，评估是否接入 ViTPose 改善人体框、多人检测及 2D 关键点/置信度。
- 基于部署环境验证各模型在同一进程/同一 GPU 中的加载顺序、显存占用和请求并发行为，再决定是否需要推理串行化或显存卸载策略。

### 本计划不包含

- 新增独立 SAM3DPose HTTP 服务或让 ground_pipeline 通过 HTTP 调用另一个姿势服务。
- 改变现有地面分割、DepthPro、RANSAC、世界坐标系或接触区算法。
- 将 SAM3DPose 姿势坐标强行转换为地面世界坐标，或据此计算脚底接触、穿透、物理约束。此类融合需要单独标定与验证。
- P0 多人姿势识别、实例级 mask/pose 匹配；第一阶段假设输入主要针对单人，验证通过后再评估是否集成 ViTPose。
- 第一阶段集成 ViTPose 检测或运行 ViTPose 2D 姿态推理；是否接入留待 P0 计划完成后的效果与资源评估。

## 3. 目标流程

```text
输入图片
  ├─ 现有地面链路：LangSAM 地面 mask → DepthPro 深度 → 反投影/RANSAC
  │                  └─ 可选 LangSAM 人物 mask → 人物点云/地面接触区
  └─ pose_inference=true 时：person_mask 像素范围 → 加边界扩展的 2D bbox → SAM3DPose Body 推理（不运行 ViTPose）

地面结果与 pose 状态分别组装 → 返回 /v1/ground
```

建议在同一次请求的地面几何处理完成后调用姿势模块，使地面链路仍是主流程；图片仅在当前进程内传入，按请求同步执行。

## 4. 请求与响应契约（拟议）

### 请求新增字段

| 字段 | 类型 | 建议默认值 | 行为 |
|---|---|---:|---|
| `pose_inference` | bool | `false` | `false` 时跳过姿势模块；`true` 时运行姿势推理。 |
| `pose_want_overlay` | bool | `false` | 是否要求姿势模块生成 overlay；默认关闭以控制响应体积。仅在启用姿势推理时生效。 |
| `pose_want_mesh` | bool | `false` | 与 `visualize=true` 同时开启时，将 SAM3D 网格独立多视角图放入 `visualization.pose_mesh`，并将完整预测网格经关键点 Umeyama 配准后加入 `visualization.views` 双面板与 `visualization.pose_world`；配准用 MHR-70 关键点在 DepthPro 深度图采样解算平移/旋转/尺度，仅修正整体位置，不修正姿势误差；配准失败自动退回原始位置。网格生成/绘制失败写入 `pose.mesh_error`，不影响地面和姿势结果。 |

开启姿势推理时使用当前进程已解码的 RGB 图像直接调用姿势模块，不经过 HTTP，不需要序列化或传输图像。

### 新增响应对象

仅在 `pose_inference=true` 时返回 `pose` 对象，避免默认响应增加体积。建议字段：

| 字段 | 说明 |
|---|---|
| `status` | `success` / `error`；未开启推理时不生成 `pose` 对象。 |
| `engine` | 姿势引擎名，预期为 `sam3d_body_vith`。 |
| `skeleton_source` | 姿势模块实际采用的骨架来源，如 `MHR127_direct` 或 `coco17_smpl24`。 |
| `joints_3d` | 姿势模块归一化后的 SMPL-24 关节位置。保留模块原值。 |
| `confidence`、`valid_joints` | 若 SAM3DPose 本身提供，则返回其关节置信度与有效关节数；不在 P0 为生成这些字段而运行 ViTPose。 |
| `keypoints_2d_coco17`、`keypoints_3d_coco17` | 仅在不依赖 ViTPose 且可由 SAM3DPose 直接提供时返回；坐标结构与单位遵循参考模块定义。 |
| `orientation_mode`、`camera`、`time_ms` | 姿势模块提供且确有下游价值的元信息。 |
| `visualization.pose_overlay` | 已由 P1 实现为独立 PNG data URI；仅在 `pose_inference=true`、`pose_want_overlay=true`、`visualize=true` 且姿势推理成功时返回，不放入 `pose`。 |
| `alignment` | 已实现；`pose_want_mesh=true` 且 `visualize=true` 时返回网格配准结果（`status`、`scale`、`rmse_m`、`keypoints_used`、`translation_m`）；`skipped` 时网格保持原始位置。 |
| `error` | `status=error` 时的错误摘要。 |

字段的最终筛选以最小可用响应为准；默认不转发 `want_raw` 数据，避免回传大量诊断数组。地面响应既有顶层字段和 `camera`/`world` 分组保持不变。

### 坐标系规则

- 地面服务现有 `camera`/`world` 几何仍使用原有相机系/地面世界系定义。
- 参考 SAM3DPose 模块的 `joints_3d` 为 **Figure 系、米**；`keypoints_3d_coco17` 为模块定义的相机系、米；2D 关键点为图像像素坐标。
- 必须在 `pose` 子对象及 API 文档中标明坐标系与单位，不得将 `joints_3d` 当作 DepthPro 反投影点，也不得未经验证套用 ground 的相机到世界变换。
- `pose` 是独立推理结果，不表示姿势已与 LangSAM 人物 mask 完成实例级匹配；第一阶段只将人物 mask 用于计算单一输入 bbox。

## 5. 处理与异常规则

1. **请求未启用或无有效人物 mask**：不运行本次姿势推理；现有地面结果照常返回。姿势模型是否在进程启动时预加载由模块配置决定，与单次请求开关分开控制。
2. **模型未就绪或推理异常**：姿势子结果为 `status=error`，地面成功结果照常返回；日志包含请求标识和阶段，不记录图片内容。
3. **CUDA 显存不足/模型运行异常**：姿势子结果标记失败；按参考适配器约定清理 CUDA 缓存（如适用），不自动无限重试。
4. **姿势业务失败**：SAM3D 未检测到人物或推理返回失败时，保留错误摘要；地面流程不回滚。
5. **姿势输出缺字段或形状不符合契约**：只将姿势部分标记失败并记录校验错误；不把不完整数据作为成功姿势返回。
6. **输入 mask 无效**：若 mask 为空或 bbox 无有效面积，不调用姿势模型，并返回明确的姿势跳过/失败状态；地面结果照常返回。
7. **输入图片问题**：图片在流水线入口统一解码；解码失败仍按现有地面 API 错误处理，不单独运行姿势模块。

## 6. 配置、就绪与资源约束

- SAM3DPose 作为 ground_pipeline 进程内模块，通过配置引用 ImagePose_Resources 中 SAM3D 所需的模型、vendor 依赖及适配器资源；第一阶段不加载 ViTPose 权重，具体资源布局需在实现阶段核对，避免复制权重。
- ground_pipeline 启动时按配置加载 SAM3D 姿势模块；姿势功能关闭时不要求姿势模型就绪，也不影响地面 `/health`。姿势功能启用但资源/CUDA 缺失时，应明确报告模块不可用，地面基础能力仍可使用。
- SAM3DPose 需要 CUDA；现有 ground 启动脚本中的 WSL CUDA 初始化顺序约束（DepthPro 先于 LangSAM）不得改动。将 SAM3D 纳入同进程后，需实测模型加载顺序、同卡显存、请求串行化及不同 CUDA 库之间的兼容性。
- 推理显存不足或 CUDA 异常时按姿势失败降级，不影响地面计算；P0 不承诺 GPU 资源隔离，也不新增未经测量的吞吐或时延指标。

## 7. 实施阶段与交付

| 阶段 | 工作 | 交付/完成条件 |
|---|---|---|
| 0. 模块契约核对 | 核对 SAM3D 的输入/输出结构、依赖与权重布局、模型版本及 CUDA 加载顺序。 | 明确可复用的模块入口和资源路径；未知依赖保持待定。 |
| 1. SAM3D 模块集成 | 在 ground_pipeline 进程内增加 SAM3D 适配、模型加载及姿势输出字段校验；用人物 mask 生成单一 2D bbox。 | 不加载 ViTPose；姿势模型可按模块配置加载复用；无效 mask、模块未就绪或输出不符合契约时，不影响地面结果。 |
| 2. 流水线编排 | 为 `/v1/ground` 增加可选开关；把 RGB 与 mask-derived bbox 交给进程内 SAM3D 模块；将姿势结果独立并入响应。 | 关闭功能时回归响应兼容；开启时地面与姿势可独立成功/失败。 |
| 3. 文档与部署 | 更新 `GROUND_SERVICE_API.md`、项目介绍及启动脚本/配置说明。 | 参数、响应结构、错误降级、坐标系及 SAM3D 模型依赖说明一致。 |
| 4. P0 验证与评估 | 先验证单人、无人物/无效 mask、遮挡、模块未就绪及 GPU 显存不足；记录资源和质量表现。P0 完成后再评估是否接入 ViTPose。 | 单人 P0 验收通过；基于实际效果和资源数据决定是否进入 ViTPose 后续评估，不将多人支持作为 P0 门槛。 |
| 5. P1 姿势可视化 | 在现有可视化响应中添加独立的可选 `visualization.pose_overlay` 2D 姿势 PNG 和 `visualization.pose_mesh` 3D 网格多视图 PNG；前者由 SAM3D 原生 MHR-70 2D 关键点绘制，后者由 `pred_vertices` 与模型拓扑绘制。 | 对应开关启用、姿势推理成功时各自独立返回；绘图失败分别写入 `pose.overlay_error` / `pose.mesh_error`，不影响姿势关节和地面结果；原 `visualization.overlay` 与 `views` 含义不变，不依赖 ViTPose。 |

## 8. 验收要点

- 默认请求不运行 SAM3DPose 模块，现有 `/v1/ground` 结果字段与含义不变。
- 开启 `pose_inference` 后，地面与姿势推理复用同一进程中的 RGB 图像；成功结果按约定返回关节、置信度和来源信息。
- 姿势模型未就绪、CUDA 异常、推理失败或输出不完整时，地面结果仍可成功返回，并明确标记姿势失败。
- 返回数据明确区分 Figure 系 `joints_3d` 与 ground 几何坐标；不产生未经验证的坐标转换。
- 姿势 overlay 和 3D 网格可视化默认关闭；请求条件满足时分别作为独立 PNG data URI 放在 `visualization.pose_overlay` 与 `visualization.pose_mesh`，不改变既有地面 `overlay` 与 3D `views`；姿势绘图失败分别报告，不丢失有效姿势 3D 关节；raw 大数组不进入常规 ground 响应。
- 地面服务健康状态按现有逻辑报告基础地面模型状态；若姿势模块配置为必需项，其加载状态需单独报告。可选姿势模块未就绪不能使地面基础能力误报不可用。

## 9. P0 完成后的评估项

第一阶段完成并有质量/资源实测结果后，再评估 ViTPose 是否值得接入；评估维度包括 bbox 改善程度、是否需要额外 2D 关键点/置信度、多人物检测需求、增加的模型依赖与显存/耗时。评估后另行决定是否排入后续阶段，不预设必须集成。

## 10. 待实现前确认的事项

1. `pose_inference`、`pose_want_overlay` 的最终字段名、SAM3D 模块启用配置及是否设为启动必需项，需在实现开始时与现有调用方约定。
2. 同一 GPU 上地面模型与 SAM3D 的显存安全边界、可接受请求并发和是否需要串行化，必须通过目标部署机实测确定。
3. `visualization` 中可选的 `pose_overlay` 由 SAM3D 原生 MHR-70 2D 图像像素坐标关键点生成；姿势推理成功但绘图失败时保留姿势 3D 结果，并在 `pose.overlay_error` 记录错误。

## 11. 参考资料

- `ground_pipeline/ground_service.py`：现有 FastAPI 请求编排、地面/人物 mask、坐标系与响应组织。
- `ground_pipeline/GROUND_SERVICE_API.md`：当前 `/v1/ground` 对接契约。
- `ground_pipeline/ground_pipeline 项目介绍.md`：现有系统流程与 CUDA 加载顺序约束。
- `G:\UGit\QingGeTools\ImagePose_Resources\SAM3D_POSE_API.md`：参考姿势模块的输入输出定义与结果语义（仅用于复用/适配参考，不作为新增 HTTP 接口）。
- `G:\UGit\QingGeTools\ImagePose_Resources\ai_vitpose\core\engine.py`、`sam3d_adapter.py`：推理编排、姿势结果和单人输出的实现参考。
- `G:\UGit\QingGeTools\ImagePose_Resources\DEPENDENCIES.json`、`MANIFEST.json`：资源包运行依赖与独立服务边界。
