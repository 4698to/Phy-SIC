# ground_pipeline：提高 SAM-3D 与图片姿势吻合度（四肢 / 脚掌 / 手指）

## 0. 结论摘要

1. **先分组量化，再改链路。** 当前 2D overlay 只画 COCO-17（[ground_pose.py:35-51](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_pose.py#L35-L51)），脚趾和手指根本不在图上。看不到误差来源就改姿态，容易把全局配准问题当成模型问题。
2. **最高性价比不在姿态优化，而在输入与全局配准。** 当前三处明显可改：
   - bbox 固定外扩 20%/15%，人物在模型输入中占比偏小（[ground_pose.py:124-141](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_pose.py#L124-L141)）；
   - `use_mask=False`，已有的 person mask 没有作为 prompt 使用（[ground_pose.py:197-206](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_pose.py#L197-L206)）；
   - `cam_int` 只在 `want_mesh=True` 时传入（[ground_pose.py:201-204](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_pose.py#L201-L204)），只出姿势时模型自己估焦距。
3. **全局配准不应让手、脚、手指参与。** 70 个关键点全部参与 Umeyama/ICP（[ground_service.py:768-777](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L768-L777)），而脚趾/手指位于 mask 边缘，DepthPro 深度噪声大，会把整体配准带偏。
4. **局部修正要受限。** 不要重开 137 维关节 + 尺度的全局优化，该方案已在实机验收中被否决（见 [验收失败记录_SAM3D姿势优化.md](file:///g:/UGit/Phy-SIC/ground_pipeline/验收失败记录_SAM3D姿势优化.md)）。正确做法是分链、少自由度、2D 驱动、冻结躯干与尺度。
5. **手指有数据，但有硬上限。** MHR-70 每手含 20 个手指点 + 1 个手腕点，MHR-127 含完整手指关节。但如果手在图中只有几十像素，或输出网格的手指本身不可动，后处理无法恢复。

推荐顺序：

```
P0 分组诊断
  → P1 输入与全局配准（不改模型）
  → P2 地面内 x/y + 四肢受限 IK + 脚掌
  → P3 手指 / 独立 2D 观测
```

---

## 1. “吻合度”拆成三层

| 层级 | 表现 | 主要来源 | 解决方向 |
|---|---|---|---|
| L-A 全局刚性 | 整个人偏移、偏大偏小、朝向不对 | bbox、焦距、DepthPro 配准、地面锚定 | 输入 + 全局配准 |
| L-B 关节角 | 肘/膝/腕/踝弯曲角度不对 | SAM3D 回归、遮挡、输入分辨率 | 受限局部 IK |
| L-C 表面细节 | 脚掌翻转、脚趾/手指不对 | 像素分辨率、足/手自由度、网格是否可动 | 裁剪 + 足/手局部修正 |

三层要分开评估。L-A 没修好时，L-B/L-C 的指标没有意义。

---

## 2. 当前链路与可改点

### 2.1 输入侧

| 位置 | 现状 | 影响 |
|---|---|---|
| [ground_pose.py:124-141](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_pose.py#L124-L141) | mask bbox 外扩 `max(20px, 20%宽)` / `max(20px, 15%高)` | 外扩过大时人物像素占比下降，手指/脚趾最先丢失 |
| [ground_pose.py:197-206](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_pose.py#L197-L206) | `use_mask=False`，只传 bbox | 没有利用 person mask 的边界/遮挡信息 |
| [ground_pose.py:201-204](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_pose.py#L201-L204) | `cam_int` 仅在 `want_mesh=True` 时传入 | 只出姿势时焦距由模型估计，尺度可能偏 |
| [ground_pose.py:172-178](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_pose.py#L172-L178) | `human_segmentor=None`、`fov_estimator=None` | 是否支持 mask prompt / 真实 FOV 需要确认 vendor 行为 |
| [ground_pose.py:238-260](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_pose.py#L238-L260) | 输出 MHR-70 的 2D 点与相机系 3D 点 | 手/脚点可用，但未分组使用 |

### 2.2 全局配准侧

| 位置 | 现状 | 影响 |
|---|---|---|
| [ground_service.py:320-355](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L320-L355) | 70 点逐点采 DepthPro 深度，权重按 mask 占比与深度离散度 | 手脚边缘权重低但非零，仍参与拟合 |
| [ground_service.py:768-777](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L768-L777) | 所有有效深度点送入后续 Umeyama / 残差 | 手指/脚趾噪声会拖动全局解 |
| [ground_service.py:807-846](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L807-L846) | 地面锚定成功时跳过相似变换，`scale=1` | 尺度不漂，但若 bbox/焦距有偏，无法自纠 |
| [ground_service.py:847-889](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L847-L889) | 地面锚定失败时走加权 Umeyama | 手/脚点会参与尺度与旋转估计 |
| [ground_service.py:727-743](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L727-L743) | 只输出残差最大的 12 个点 | 没有按 body/hand/foot 分组统计 |

### 2.3 精修侧

| 位置 | 现状 | 影响 |
|---|---|---|
| [ground_pose.py:382-406](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_pose.py#L382-L406) | 平移只沿地面法向施加 | 画面内左右/前后偏移无法被 2D 投影 loss 修正 |
| [ground_pose.py:279-286](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_pose.py#L279-L286) | 接触 200、穿透 400、投影 150 | 接触项主导，接触点选错时会牺牲 2D 一致性 |
| [ground_pose.py:320-341](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_pose.py#L320-L341) | 最低 60 个顶点定义接触高度 | 不能区分脚掌、臀部、手撑地 |
| [ground_pose.py:412-429](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_pose.py#L412-L429) | 接触/穿透/Chamfer/投影联合优化 | 全部作用于整体刚体，不改关节角 |

---

## 3. P0：分组诊断（先做，不改行为）

### 3.1 增加 70 点分组 overlay

现有 overlay 只取 17 个 COCO 点。建议新增一张调试图，按组着色：

| 组 | MHR-70 索引 | 点数 |
|---|---|---|
| body | 0-14 | 15 |
| foot | 15-20 | 6 |
| right hand | 21-41 | 21 |
| left hand | 42-62 | 21 |
| body landmarks | 63-69 | 7 |

每组用不同颜色、不同点径；手掌/脚掌单独画放大子图。这样可以直接看出是“整条手臂角度错”还是“只有指尖错”。

### 3.2 指标

按组统计，不要只看全局 RMSE：

| 指标 | 定义 | 用途 |
|---|---|---|
| `rmse_body_px` | body 组 2D 重投影 RMSE | 判断全局配准 |
| `rmse_hand_px` | 左右手合计 RMSE | 判断手腕/手指 |
| `rmse_foot_px` | 脚 6 点 RMSE | 判断踝/脚掌 |
| `rmse_finger_px` | 指尖 + 指节 RMSE | 判断手指 |
| `hand_span_px` | 手部关键点包围盒最长边 | 判断是否值得优化手指 |
| `foot_span_px` | 脚部 6 点包围盒最长边 | 判断脚部是否有足够像素 |
| `sole_angle_deg` | 脚底拟合平面与地面夹角 | 判断脚掌翻转 |
| `sole_gap_m` | 脚底最低点到地面距离 | 判断脚掌悬空/穿透 |

归一化建议：像素误差除以人物像素高度，跨图片可比。

### 3.3 误差归因

| 观察 | 判断 | 优先动作 |
|---|---|---|
| body 组整体同向平移 | 全局配准/相机中心偏 | P1 全局配准 |
| body 组误差随距离增大 | 尺度/焦距偏 | P1 传真实 K、调 bbox |
| body 组好，hand/foot 差 | 局部姿态 | P2 局部 IK |
| 手/脚误差大且 `span_px` 小 | 分辨率上限 | P1 裁剪，暂不做局部 IK |
| 脚掌角度错但脚踝位置对 | 踝关节/脚底自由度 | P2 脚掌 |
| 手指全错且指尖粘连 | 网格手指不可动 | P3 外挂手部模型 |

---

## 4. P1：输入与全局配准（不改模型）

### 4.1 全局配准只用躯干与大关节

**做法**

- 定义 `GLOBAL_KP = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 69]`，可选加 63-68。
- 脚 15-20、手 21-62 不参与 Umeyama、ICP、depth 目标，只参与评分。
- 若有效 body 点少于 8 个（严重遮挡/出画），回退到现有全点逻辑。

**改动点**

- [ground_service.py:768-777](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L768-L777)：在进入 `_keypoints_depth_camera` 前先按索引分组。
- [ground_service.py:320-355](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_service.py#L320-L355)：为手/脚点单独计算权重与诊断，不进入全局拟合。

**预期**

- 消除手指/脚趾深度噪声对全局旋转、尺度的影响。
- 对“整个人略微歪/偏”的改善通常比改姿态更明显。

**风险**

- 纯站立、四肢舒展且 body 点被遮挡时，可用点变少。用 8 点下限 + 回退兜底。

### 4.2 `cam_int` 始终传入

**做法**

- 去掉 `want_mesh` 门控，只要 `cam_int` 存在就传入。
- 在返回的 `camera.calibration` 中区分 `measured_intrinsics` 与 `model_default_fov_not_measured`。

**改动点**

- [ground_pose.py:201-204](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_pose.py#L201-L204)。

**预期**

- 尺度、远近、肢体投影更稳定，尤其是人物不在地面锚定路径时。

**风险**

- 如果 SAM3D 内部默认 FOV 与 DepthPro 的 K 差异较大，输出尺度会变化。用分组 `rmse_*_px` 做 A/B，不要凭目视切换。

### 4.3 bbox、裁剪与有效分辨率

**做法**

1. 先做 bbox 外扩 A/B：`5% / 10% / 20%`，按 `rmse_body_px` 选择。
2. bbox 补成正方形，避免模型输入各向异性缩放。
3. 确认 SAM3D 的实际输入分辨率。如果内部会缩到较小尺寸，则改为“按人物裁剪后再送入”，保证人物高度在模型输入中足够大。
4. 裁剪模式下需要同步重映射：
   - `K' = [[fx*s, 0, (cx-x0)*s], [0, fy*s, (cy-y0)*s], [0,0,1]]`
   - 2D 关键点回原图：`u = u'/s + x0`，`v = v'/s + y0`
   - 3D 相机系点不需要加裁剪偏移，但 focal 输出要除以 `s`。

**改动点**

- [ground_pose.py:124-141](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_pose.py#L124-L141)：bbox 生成。
- [ground_pose.py:181-206](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_pose.py#L181-L206)：裁剪、K 重映射、结果回映射。

**预期**

- 手指/脚趾的可用像素显著增加，是 L-C 层最有效的一步。

**风险**

- 裁剪过紧会切掉脚尖/指尖；必须保留至少 5% 边距。
- 模型若使用固定输入尺寸，裁剪只提升有效分辨率，不改变网络输入尺寸；需要实测确认收益。

### 4.4 mask prompt

**做法**

- 先核对 vendor 的 `process_one_image` 是否接受 `masks` 参数。
- 若支持，传 `masks=person_mask, use_mask=True`；若不支持，保持 `use_mask=False`，只做 bbox 优化。

**改动点**

- [ground_pose.py:197-206](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_pose.py#L197-L206)。

**预期**

- 减少背景/遮挡物对四肢末端的干扰，边界更贴合。

**风险**

- 当前 `human_segmentor=None`（[ground_pose.py:172-178](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_pose.py#L172-L178)），开启后可能要求外部 mask。必须做异常回退，不能因 mask 失败导致整条姿势链路不可用。

### 4.5 测试时增强（可选质量档）

**做法**

- 水平翻转 + 2 个尺度，各跑一次，关节/顶点取中位数或平均。
- 只在“高精度模式”开启，默认关闭。

**预期**

- 对手指、脚趾这类小结构最有效，能降低单次回归抖动。

**风险**

- 推理时间约 2-3 倍；需要确认显存余量（当前 12GB 卡约留 4GB）。

---

## 5. P2：后处理（局部修正）

### 5.1 地面平面内 2-DOF 平移

**问题**

当前 E 段平移只沿地面法向（[ground_pose.py:382-406](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_pose.py#L382-L406)）。投影 loss 虽然存在，但 x/y 动不了，画面里的左右/前后偏移修不回来。

**做法**

- 在 D 段落位之后，增加一个仅地面平面内 x/y 的 2-DOF 微调。
- 目标函数：2D 关键点重投影误差（body 组权重高，手/脚低）。
- 约束：`|Δxy| ≤ 3cm`；z 不变；接触高度不变。
- 微调后重新检查脚/臀接触间隙，若恶化则回退。

**预期**

- 直接改善“整体和图片对不上”，不影响接触，不改姿势。

**风险**

- 若 DepthPro 地面法向有偏，x/y 与 z 不完全解耦；用地面平面投影而不是世界坐标硬编码。

### 5.2 四肢受限 IK

**原则**

- 只优化四肢链，不放开躯干、root、体型、尺度。
- 只用 2D 重投影，不用 DepthPro 深度项。
- 分链求解，单链失败不影响其他链。

**自由度建议**

| 链 | 关节 | 自由度 | 单次上限 | 累计上限 |
|---|---|---|---|---|
| 手臂 | 肩 3 + 肘 1-2 + 腕 2 | 6-7 | 5° | 10-15° |
| 腿 | 髋 3 + 膝 1 + 踝 2 | 6 | 5° | 10-15° |

优先匹配：肘、腕、膝、踝；肩/髋只在残差持续偏大时小幅放开。

**前提**

- 需要 MHR 前向/蒙皮，把 127 个关节旋转作用回网格。
- 当前 checkout 未包含 `vendor/sam-3d-body`；需先确认 `pred_joint_global_rots` 与 skinning weights 是否可访问。
- 若只能拿到关节坐标、拿不到可微前向，则降级为“关节级 IK + 重定向”，效果会打折。

**预期**

- 改善肘/膝/腕/踝的弯曲方向和角度，尤其是遮挡侧。

**风险**

- 单视图 2D 约束存在深度歧义：肢体可以转到画面外仍然投影正确。必须加关节角上限、骨长保持、弱深度/先验正则。
- 若残差本来就小（< 5px），不要强行优化，避免把好结果改坏。

### 5.3 脚掌

**问题**

当前用“全局最低 60 个顶点”定义接触（[ground_pose.py:320-341](file:///g:/UGit/Phy-SIC/ground_pipeline/ground_pose.py#L320-L341)），无法区分脚掌、臀部、手撑地。

**做法**

1. 用 MHR-70 的 15-20（toe/heel 6 点）和 MHR-127 的 `l_foot / l_ball / r_foot / r_ball` 定位脚部。
2. 通过蒙皮权重选出“以脚部关节为主”的顶点；没有权重时用脚踝附近几何邻域近似。
3. 对脚部顶点拟合脚底面，约束：
   - 脚底面与地面夹角 ≤ 5°；
   - 脚底最低点接触间隙 `|gap| ≤ 1cm`；
   - 踝关节 pitch/roll 单次 ≤ 5°，累计 ≤ 10°；
   - 脚趾弯曲只在 toe 关键点残差大时放开，≤ 10°。
4. 脚掌接触与身体其他接触分开统计，避免手撑地被误判为脚掌。

**预期**

- 改善脚掌翻转、脚尖翘起、脚跟悬空。

**风险**

- 鞋子/厚底会让“脚底平面”不等于网格最低面；需要用 mask 或深度确认脚底区域。
- 脚部顶点选择错误会把小腿拉进平面拟合；先做可视化验证。

### 5.4 手指

**先判断值不值得做**

| 条件 | 处理 |
|---|---|
| `hand_span_px ≥ 80` 且网格手指可动 | 做逐指 2D IK |
| `hand_span_px < 80` | 不优化手指，只保证手腕位置 |
| 网格手指融合/刚性 | 后处理无效，转 P3 外挂手部模型 |

**做法**

- MHR-70：每手 20 个手指点（4 点/指）+ 1 个 wrist 点，作为 2D 目标。
- MHR-127：`thumb0-3 / index1-3 / middle1-3 / ring1-3 / pinky1-3`，作为关节变量。
- 每根手指一条链，解析 2D IK 或小规模数值 IK；不全局联合优化。
- 只改手指关节，不改手掌朝向、腕位置、前臂。
- 单关节 ≤ 10°，指尖位移上限 2-3cm。

**预期**

- 张开手、比手势、手扶物体等场景改善明显。

**风险**

- 手指像素太少时，2D 目标本身不可信，优化会放大噪声。
- 若 MHR 网格手指没有独立蒙皮，关节旋转不会改变顶点，必须外挂 WiLoR/MANO。

---

## 6. P3：独立 2D 观测

SAM3D 的 70 点是自己预测的，用自己的点优化自己会“自洽但不准”。要突破上限，需要独立观测：

| 部位 | 观测源 | 说明 |
|---|---|---|
| 身体 | ViTPose 17 点 | 仓库已有，但 ground_pipeline P0 刻意未接 |
| 手 | MediaPipe Hands / WiLoR 21 点 | 独立手部关键点，适合手指 |
| 脚 | SAM3D toe/heel 或分割轮廓极值 | 脚部独立检测器收益通常小于手 |

原则：

- 独立 2D 点只作为目标，不直接替换 SAM3D 输出。
- 只在高精度模式启用，默认关闭。
- 独立点与 SAM3D 点残差过大时，按置信度融合，不直接采纳。

代价：额外模型、显存、延迟。需要重新评估 12GB 卡余量。

---

## 7. 不推荐的做法

1. **重开 137 维关节 + 尺度全局优化。** 已被实机否决：`delta_pose_rms=0.0508 rad`、`max_vertex_shift=0.133m`、2D 残差 `0.2642→0.2845m`。见 [验收失败记录_SAM3D姿势优化.md](file:///g:/UGit/Phy-SIC/ground_pipeline/验收失败记录_SAM3D姿势优化.md)。
2. **只用 2D 重投影、无先验地优化四肢。** 单视图深度歧义会导致肢体转到画面外仍然投影正确。
3. **用手指/脚趾关键点参与全局配准。** 边缘深度噪声会拖动整体尺度与旋转。
4. **在小手/小脚上强行优化。** 像素不足时只会拟合噪声。
5. **用 2D overlay 的 17 点判断手指/脚趾。** 当前 overlay 根本不画这些点。

---

## 8. 验收指标与回归集

### 8.1 指标

| 指标 | 建议阈值 | 说明 |
|---|---|---|
| `rmse_body_px_norm` | 不高于基线 | 全局配准不得退化 |
| `rmse_hand_px_norm` | 相对基线下降 ≥10% | 手部整体 |
| `rmse_foot_px_norm` | 相对基线下降 ≥10% | 脚部整体 |
| `rmse_finger_px` | 仅在 `hand_span_px ≥ 80` 样本统计 | 手指 |
| `sole_angle_deg` | ≤ 5° | 脚底与地面 |
| `sole_gap_m` | `|gap| ≤ 1cm` | 脚底接触 |
| `global_xy_shift_m` | ≤ 3cm | 地面内微调上限 |
| `joint_delta_deg` | 单关节 ≤ 5-10° | 局部 IK 上限 |
| `max_vertex_shift_m` | 局部修正 ≤ 3-5cm | 防止改写姿势 |
| `silhouette_iou` | 不下降超过 3% | 轮廓一致性 |
| `hand_span_px` | 记录并分桶 | 判断手指结论是否可信 |

### 8.2 回归集

至少覆盖：

- 站立、双手自然下垂（基线）
- 站立、张开手掌/比手势（手指）
- 蹲姿、脚掌完全着地（脚掌）
- 坐姿、手撑地/手扶膝（手部与接触分离）
- 四肢被躯干遮挡的侧视角（遮挡侧）
- 人物只占画面很小比例（分辨率下限）
- 脚部被鞋子/裙摆遮挡（脚底平面失败）

### 8.3 验收方式

1. 分组 2D overlay（70 点，按组着色）。
2. 原图叠加网格轮廓，2 倍放大看手/脚。
3. 三视图网格 + 地面。
4. 指标表按样本分桶，不只看平均值。
5. 任何一项超限即回退该子功能，不整体回滚。

---

## 9. 风险与回退

| 风险 | 触发条件 | 回退 |
|---|---|---|
| vendor 不支持 mask prompt | `process_one_image` 无 `masks` 参数 | 保持 `use_mask=False`，只做 bbox/裁剪 |
| 裁剪后 2D/K 回映射错误 | 输出整体偏移 | 关闭裁剪，回退原图推理 |
| 真实 K 与模型默认 FOV 冲突 | 尺度明显变化 | 按 `rmse_body_px` A/B，保留更优路径 |
| 四肢 IK 造成自穿透/关节反折 | 残差下降但目视异常 | 关闭 IK，保留 P1 |
| 手指网格不可动 | 关节转动顶点不变 | 关闭手指修正，转外挂手部模型 |
| 脚部顶点选错 | 小腿被拉入脚底平面 | 关闭脚掌修正，回退全局接触 |
| 延迟/显存超限 | TTA 或独立 2D 模型 | 默认关闭，仅高精度模式启用 |

所有新功能建议用请求字段控制，默认关闭或保持现有行为：

```json
{
  "contact_refine": false,
  "pose_refine": {
    "enable": false,
    "limb_ik": false,
    "foot": false,
    "fingers": false,
    "max_joint_delta_deg": 5,
    "max_vertex_shift_m": 0.05,
    "max_xy_shift_m": 0.03
  }
}
```

响应中增加分组诊断，便于线上定位：

```json
{
  "pose_fit": {
    "rmse_body_px_norm": 0.0,
    "rmse_hand_px_norm": 0.0,
    "rmse_foot_px_norm": 0.0,
    "rmse_finger_px": null,
    "hand_span_px": 0,
    "foot_span_px": 0,
    "sole_angle_deg": null,
    "sole_gap_m": null,
    "applied": {
      "global_xy": false,
      "limb_ik": false,
      "foot": false,
      "fingers": false
    }
  }
}
```

---

## 10. 实施顺序与门禁

### 阶段

| 阶段 | 内容 | 改动量 | 依赖 |
|---|---|---|---|
| P0 | 70 点分组 overlay、分组指标 | 小 | 无 |
| P1.1 | 全局配准排除手/脚 | 小 | 无 |
| P1.2 | `cam_int` 始终传入 | 小 | 无 |
| P1.3 | bbox A/B、裁剪与 K 重映射 | 中 | 确认模型输入尺寸 |
| P1.4 | mask prompt（条件启用） | 小 | 确认 vendor 签名 |
| P2.1 | 地面内 x/y 2-DOF | 中 | 无 |
| P2.2 | 四肢受限 IK | 大 | MHR 前向/蒙皮 |
| P2.3 | 脚掌 | 中 | 脚部顶点选择 |
| P2.4 | 手指 | 大 | 手指可动 + 足够像素 |
| P3 | 独立 2D 观测 | 大 | 额外模型/显存 |

### 门禁

- **G0（诊断）**：分组 overlay 与指标可用，能稳定区分 L-A / L-B / L-C。
- **G1（输入与全局）**：`rmse_body_px_norm` 不退化，手/脚指标有可量化改善；接触与穿透指标不变差。
- **G2（局部）**：四肢/脚掌在回归集上改善，且 `max_vertex_shift`、`joint_delta`、`silhouette_iou` 不超限。
- **G3（手指）**：仅在 `hand_span_px ≥ 80` 的样本上评估；不达标就关闭手指功能。

任何阶段未过门禁，不进入下一阶段。

---

## 附录：MHR-70 关键点索引

| 索引 | 名称 | 分组 |
|---|---|---|
| 0 | nose | body |
| 1-4 | eyes / ears | body |
| 5-6 | shoulders | body |
| 7-8 | elbows | body |
| 9-10 | hips | body |
| 11-12 | knees | body |
| 13-14 | ankles | body |
| 15-17 | left big toe / small toe / heel | foot |
| 18-20 | right big toe / small toe / heel | foot |
| 21-40 | right hand：thumb/index/middle/ring/pinky 各 4 点 | hand |
| 41 | right wrist | hand |
| 42-61 | left hand：thumb/index/middle/ring/pinky 各 4 点 | hand |
| 62 | left wrist | hand |
| 63-64 | olecranon | body landmark |
| 65-66 | cubital fossa | body landmark |
| 67-68 | acromion | body landmark |
| 69 | neck | body landmark |

说明：索引与名称来自 [mhr_source_profile.v1.json:35-52](file:///g:/UGit/QingGeTools/ImagePose_Resources/assets/mhr/profile/mhr_source_profile.v1.json#L35-L52)；MHR-127 的完整手指关节名见同文件 15-34 行。
