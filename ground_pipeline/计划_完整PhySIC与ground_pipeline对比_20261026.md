# 计划：直接部署完整 PhySIC 与现有 ground_pipeline 的对比

- 文档日期：2026-10-26
- 文档性质：路线对比与决策计划（不含代码改动）
- 对比对象：A = 本仓根目录完整 PhySIC（`run_optimizer.py` 三阶段）；B = `ground_pipeline/` 现有在线地面 + 姿势服务
- 结论口径：可核对的事实一律标注来源文件与行号；不可核对的一律标「待实测」，不填空数字

---

## 0. 结论摘要（TL;DR）

| 问题 | 结论 |
|---|---|
| 在现有机器（RTX 4070 / 12282 MiB ≈ 12 GB）上「直接部署完整 PhySIC」可行吗？ | **不具备可用性**。`INSTALL.md` 明写需要 ≥40 GB 显存（原因是 OmniEraser 基于 FLUX）；`run_optimizer.py` 只在显存 >60 GB 时才关闭 CPU offload，12 GB 会走「全模型反复上/下卡」路径，单图耗时不可控 |
| 完整 PhySIC 能替代 ground_pipeline 吗？ | **不能直接替代**。完整版是**离线批处理 CLI**（遍历 `images/`，产出 `humanscene.ply` + `scene_data_final.pkl`），没有 HTTP 服务、没有排队、没有取消、没有健康检查；而 ground_pipeline 是线上服务（`/health`、`/v1/ground`、`/v1/pose/*`、`/v2/pose/*`） |
| 完整 PhySIC 精度更高吗？ | **大概率更高，但不可验证**。完整版多了 SMPL-X 拟合 + DECO 接触先验 + 三段联合优化；但 README 的 Roadmap 明确 **Evaluation code 未发布**，官方只给 demo。因此无法与 ground_pipeline 现成的 `rmse_m` / `tilt_deg` / `shift_m` / `chamfer` 指标同口径对标 |
| 部署成本差多少？ | 数量级差异：模型 8 个 vs 2–3 个；依赖含 pytorch3d **源码编译** + mmcv 1.3.9 钉死 + 收尾强制 `numpy==1.26.4`；数据需 **4 个学术账号**（SMPL / SMPL-X / AGORA / CameraHMR）授权 |
| 建议 | **主线不动**：线上服务继续走 ground_pipeline。完整 PhySIC 作为**独立 conda env 的离线高保真参考支线**，仅在难例复核、数据生产、指标标定时启用；不进入交付链路、不承诺在线吞吐 |

一句话：**两者不是同一类东西，是「研究级离线重建器」与「工程级在线几何服务」的区别；本机硬件也决定了完整版只能离线、且要换机器。**

---

## 1. 两条路线到底是什么

### 1.1 路线 A：完整 PhySIC（本仓根目录）

定位：SIGGRAPH Asia 2025 论文工程（arXiv 2510.11649），单张 RGB → 人体网格 + 场景表面 + 人-场景接触图。

三段主线（`optimizer.py`，`HumanScene.__init__`，L94–L436）：

| 阶段 | 做什么 | 用到的模型 |
|---|---|---|
| Stage 1 场景重建 | 人/地分割 → 背景擦除重绘 → 米制深度与点云 → 地面平面延展 | LangSAM（`sam2.1_hiera_large`）、OmniEraser（`FLUX.1-dev`）、DepthPro、MoGe v1 |
| Stage 2 人体重建 | 2D 关键点 → 相机系人体网格 → 手部精修 | ViTPose-H wholebody、CameraHMR、WiLoR-mini、SMPL-X、smplfitter |
| Stage 3 联合物理优化 | 接触先验 + 穿透约束 + 多轮联合优化 | DECO、kNN 场、Adam / L-BFGS |

三段优化（`cfg/v1.yaml`）：`opt_1` 30 步 Adam（只动 `cam_trans`）→ `opt_2` 2 步 L-BFGS → `opt_3` 100 步 Adam（放开 `scale`、`cam_trans`、`betas`、`body_pose`、`global_orient`、`left/right_hand_pose`），投影损失权重 10000、正则 45。

运行形态（`run_optimizer.py`）：

- 一次性 `load_gsam / load_vitpose / load_omni / load_chmr / load_deco / load_wilor / load_moge / load_dpro`，**8 个模型常驻**；
- `if get_gpu_memory() > 60: os.environ["DISABLE_OFFLOAD"] = "1"`，否则走 offload；
- `for image_path in image_paths:` 串行跑，每图产出 `scene_data_final.pkl`（`depth, K, pts3d, inlier_mask, scale, normals, plane_points, plane_normal, body_params, cam_trans`）与 `humanscene.ply`；
- **没有** server / 路由 / 队列 / 并发 / 超时 / 健康检查。

### 1.2 路线 B：现有 ground_pipeline

定位：面向插件与业务的生产服务，只取 PhySIC Stage 1 的「地面 + 米制锚定」能力，加上独立的姿势支线。

组成（`ground_service.py`）：

- LangSAM（GroundingDINO + `sam2.1_hiera_small`，L2308/L57）+ DepthPro（固定 1536²，L2339）；
- `extract_ground`：RANSAC 256 次 + SVD + 竖直面剔除；
- 可选 SAM3D Body ViT-H 姿势支线；
- FastAPI + Uvicorn 单进程服务，端点 `GET /health`、`POST /v1/ground`、`POST /v1|/v2/pose/single`、`POST /v1|/v2/pose/cancel`、`GET /v1|/v2/pose/health`（`GROUND_SERVICE_API.md` §3）。

运行形态：常驻服务；冷启动约 2 min（`GROUND_SERVICE_API.md` L30）；单次 `/v1/ground` 约 10–30 s（L643，明确建议调用方设 ≥120 s 超时）；与 qwen3-vl 同机约 7.8 GB 显存（瘦身方案 §八）。

### 1.3 一句话定位差异

- A 回答的是「**这张图里的整个三维世界长什么样、人和它怎么接触**」；
- B 回答的是「**地面在相机坐标系的哪个位置、人站在哪**」，其余（SMPL-X、接触图、穿模修正）整体不做。

这就是瘦身方案 L52 已经定过的取舍：「取地面为锚 + 可选姿势……Stage 2 的 SMPL-X 拟合与 Stage 3 的联合物理优化整体移除」。本文件是在同一取舍上，补一份「若反过来走完整版」的成本与风险账。

---

## 2. 能力对比

| 能力项 | 完整 PhySIC（A） | ground_pipeline（B） | 对业务的实际影响 |
|---|---|---|---|
| 人体姿势（3D） | SMPL-X 全身参数 + 手部（WiLoR），经联合优化修正 | SAM3D Body 网格 + 重定向到 `bip22_pose_v5`（可选支线） | B 的输出已直接对齐下游骨骼；A 需额外做 SMPL-X → 目标骨骼重定向 |
| 人体形状/体型 | `betas` 可优化 | 无 | 一般业务不需要 |
| 场景表面 | 人擦除后的干净场景深度点云 + 法线 + 地面延展 | 仅地面平面 + 地面点云 | A 可做遮挡区补全，B 不做 |
| 地面 | `plane_points / plane_normal`，可延展到大范围 | `plane_normal` + `plane_d`，带 `rmse_m` / `tilt` 诊断 | B 的接口契约更明确、可校验 |
| 米制尺度 | 优化变量 `scale` + DepthPro/MoGe 米制对齐 | 固定 `camera_ground_anchor`，`scale ≡ 1.0` | B 的尺度误差不进网格，历史上更稳 |
| 人-场景接触 | DECO 接触概率图 + 接触损失 | 接触报告（网格最低点/关节），无接触先验 | A 才能做「手扶桌沿」这类约束 |
| 穿透/穿模 | 显式惩罚（`max_distance_interpenetration: 0.15`） | 无 | A 用于摆拍质量，B 用于落位 |
| 多人 | 支持（`single_human: false`，逐实例） | 单人假设（瘦身方案 §九） | 多人场景 A 才有解 |
| 输出格式 | `.ply` 网格 + `.pkl` 中间量 | JSON/XML + 可视化图 + 可选 FBX 导出 | B 已接 Max 插件 |
| 服务化 | 无 | 有（含取消、排队计划） | 决定性差异 |
| 可评测性 | **Evaluation code 未发布**（README Roadmap） | 有现成基线（`rmse_m` / `tilt_deg` / `shift_m` / `chamfer`） | B 的每次改动可回归，A 只能看图 |

---

## 3. 资源与部署成本对比

| 维度 | 完整 PhySIC（A） | ground_pipeline（B） |
|---|---|---|
| 显存门槛 | **≥40 GB**（`INSTALL.md`，因 OmniEraser/FLUX） | 约 7.8 GB（与 qwen3-vl 同机，瘦身方案 §八） |
| 显存策略 | >60 GB 才 `DISABLE_OFFLOAD=1`，否则全模型 offload（`run_optimizer.py`） | 无 offload，模型常驻 |
| 模型数量 | 8 个（gsam / vitpose / omni / chmr / deco / wilor / moge / dpro） | 2–3 个（LangSAM / DepthPro / 可选 SAM3D） |
| 分段模型档位 | LangSAM `sam2.1_hiera_large` + FLUX.1-dev bf16 + CameraHMR + WiLoR + ViTPose-H + DECO + MoGe ViT-L + DepthPro | LangSAM `sam2.1_hiera_small` + DepthPro（+ SAM3D ViT-H） |
| Python / CUDA | conda `python=3.10`，CUDA 12.1（`INSTALL.md`） | 既有 `.venv-ground`（`WSL_SETUP.md`） |
| 依赖安装 | `install.sh`：torch 2.3.1+cu121、`lang-segment-anything@918043e`、WiLoR-mini、smplfitter、CameraHMR requirements、`pip install -e ml-depth-pro`、**ninja + pytorch3d 源码编译**、openmim + **mmpose v0.24.0** + **mmcv 1.3.9 `--no-deps`** + mmdet、`pip install -v -e ViTPose`、收尾强制 `numpy==1.26.4` | pip 装依赖即可；但已有 WSL 加载顺序硬约束（先深度模型后 LangSAM） |
| 授权数据 | `fetch_data.sh` 需 **4 个学术账号**：SMPL、SMPL-X、AGORA、CameraHMR，另含 DECO checkpoint / PROX | 无需账号；DepthPro 权重为 Apple ASCL（**非商业**） |
| 冷启动 | 8 模型串行加载，**未实测**（模型体量显著大于 B 的 2 min） | 约 2 min |
| 单次耗时 | 上游列写「端到端 <27 s」，但**未标注硬件条件、是否含加载**；12 GB + offload 下**必然显著劣化，未实测** | 10–30 s（长边 ≤2048） |
| 并发 | 无服务、无并发概念 | 单进程串行；排队方案见 `计划_姿势推理服务队列_20261026.md` |
| 磁盘 | FLUX + CameraHMR + DECO + SMPL-X + PROX，体量远大于 B | 约 DepthPro(2.5 GB 权重) + SAM2.1 Small |
| 可移植性 | 依赖学术授权与 40 GB 卡，换机成本高 | 现有 WSL 环境已跑通 |

**成本结论**：A 的部署不是「多装几个包」，而是「换一张 40 GB 以上的卡 + 谈 4 份学术授权 + 源码编译一套老版本 mm 生态」。B 已经在跑。

---

## 4. 精度与可验证性对比

### 4.1 A 的精度优势来自哪里（机制层面，非测量）

1. **SMPL-X 参数化 + 100 步联合优化**：mesh 会朝投影、接触、穿透三个约束收敛，而 B 的 SAM3D 网格是单次前向输出，不做闭环修正。
2. **接触先验（DECO）**：能把「脚踩地、手扶桌」这类语义约束注入几何，B 完全没有这一层。
3. **人擦除后的场景深度**：OmniEraser 先把人抹掉再算场景点云，地面拟合不会被人体点污染；B 直接在全图上做 RANSAC + 竖直面剔除，靠启发式抗干扰。
4. **多人支持**：B 是单人假设。

### 4.2 但 A 不可验证

- README 的 Roadmap 明确 **Evaluation code 未发布**，仓库只有 demo 路径（`fetch_data.sh` → `images/` → `run_optimizer.py` → `outputs/`）。
- B 有可回归的量化基线（瘦身方案 §七）：`rmse_m`、`tilt_deg`（3.47° / 1.79°）、`shift_m`（+8.99 / +2.50 cm）、`chamfer_m2`（0.0279），并能对同一批图重复跑。
- 因此**任何「A 比 B 准 X%」的说法都没有依据**。若要用 A 做精度参照，必须先自建评测集与指标脚本（见 §8 P2），这一步本身是独立工作量。

### 4.3 精度风险的不对称

| 风险 | A | B |
|---|---|---|
| 深度尺度误差 | 由 `scale` 优化变量吸收，可能整形网格（历史上 `scale=0.7878` 缩水 21% 的事故发生在 B 的回退路径） | 主路径 `camera_ground_anchor` 恒 1.0，尺度误差不进网格 |
| 依赖授权数据缺失 | 直接跑不起来（SMPL/SMPL-X 必须） | 无影响 |
| 版本漂移 | mmcv 1.3.9 + numpy 1.26.4 钉死，任何升级都可能炸 | 依赖面小 |
| 单图不稳定 | 多阶段串联，任一阶段失败即整图失败 | 单段失败可降级（`ground_snap: skipped` 等） |

---

## 5. 在 RTX 4070（12 GB）上的可行性判定

已核实事实：

- 本机 GPU：`NVIDIA GeForce RTX 4070, 12282 MiB`（实测 `nvidia-smi`）。
- `INSTALL.md`：`requires a GPU with atleast 40GB of VRAM since we use OmniEraser based on FLUX`。
- `run_optimizer.py`：只有 `get_gpu_memory() > 60` 才关 offload，12 GB 属于「offload 档」。
- `image_helpers.py`：`load_omni()` 加载 `FLUX.1-dev` transformer（bf16），`get_inpainted_image_omni()` 在启用 offload 时 `pipe.to("cuda")` 整段上卡推理，分辨率固定 1024×1024、固定种子 42。

判定：

| 问题 | 判定 | 依据 |
|---|---|---|
| 能不能装上？ | 不一定。`install.sh` 里的 pytorch3d 源码编译、mmcv 1.3.9 对 torch 2.3.1 的兼容，属于「能装但容易炸」区间 | `install.sh` |
| 能不能跑通？ | **未验证**。FLUX 在 12 GB 上通常需要 sequential offload / 分块，仓库只在 `pipe.to("cuda")` 与「全 offload」之间切换，无中间档 | `image_helpers.py` |
| 单图要多久？ | **不可承诺**。8 模型 offload + 100 步优化，量级上是「分钟到十几分钟/图」，但这是推断，必须实测 | `run_optimizer.py` / `cfg/v1.yaml` |
| 能当服务吗？ | **不能**。没有 server，且 offload 会与同机 qwen3-vl 抢显存 | `run_optimizer.py` |

**结论：12 GB 机器上的完整 PhySIC 只能定位为「离线、逐图、可长时间等待」的实验通道；在线服务不可行。**

---

## 6. 产出物与集成形态对比

| 项 | A | B |
|---|---|---|
| 调用方式 | `python run_optimizer.py`，图片放 `images/` | HTTP：`/v1/ground`、`/v1/pose/single`、`/v2/pose/single` |
| 单位 | 目录（每图一个 `outputs/<name>/`） | 请求/响应 |
| 主产物 | `humanscene.ply`、`scene_data_final.pkl` | JSON/XML + 可视化 PNG + 可选 FBX |
| 下游对接 | 需自研 PLY/参数 → 骨骼/引擎转换 | 已对接 Max 插件（`插件端对接契约.md`） |
| 失败语义 | 抛异常 / 中断整批 | 分阶段 `status`/`stage`，可降级 |
| 可观测性 | 无 | `/health`、`/v1/pose/health`、`time_ms`、可选 `/metrics`（见队列计划） |

---

## 7. 决策建议

### 7.1 推荐（默认）

**保持 ground_pipeline 为唯一在线链路，完整 PhySIC 作为离线参考支线，且不承诺交付时间。**

理由按权重排序：

1. 硬门槛：40 GB vs 12 GB，是「换机器」级别的差距，不是调参能解决的；
2. 形态不匹配：A 无服务化，要服务化等于自己写一层 FastAPI + 队列 + 产物转换，工作量不低于重写；
3. 不可评测：A 无 evaluation code，无法证明它比 B 准，无法做回归；
4. 授权链：SMPL/SMPL-X 学术授权不适合进入商业交付链路；
5. B 已被业务验证：有 API 契约、有验收基线、有失败记录与修复计划。

### 7.2 分场景建议

| 场景 | 建议 | 说明 |
|---|---|---|
| 线上插件推理（当前主业务） | **只用 B** | 延迟、并发、契约都已成型 |
| 单张素材的「高保真重建」交付 | A 离线跑，人工验收 | 接受分钟级耗时，不看吞吐 |
| 难例复盘（B 的 `rmse_m` 超阈值图） | 用 A 做对照参考，**只做定性判断** | 不做数字对标 |
| 训练/评测数据生产（带接触标注） | A 是目前唯一选择 | 接触图 + 穿透修正无法由 B 产出 |
| 多人场景 | A；或另行立项扩展 B | B 现有单人假设 |
| 需要「人扶/人靠」类物理约束 | A | B 无接触先验 |

### 7.3 不推荐的做法

- 把 A 的能力「顺手」塞进 B 的进程（显存会互相挤爆，且依赖冲突：A 钉 numpy 1.26.4 / torch 2.3.1，B 的环境未必同版本）；
- 在 12 GB 机器上对外承诺 A 的吞吐；
- 用 A 的 demo 输出比值来宣称精度提升。

---

## 8. 若决定部署完整版：共存方案（P0 → P2）

原则：**物理隔离**。独立 conda env、独立数据目录、独立 GPU 时段，不与 `ground-workspace` / `.venv-ground` 共享任何东西。

### P0 环境与数据（不碰现有服务）

1. 新建 conda env：`conda create -n physic python=3.10 -y`（`INSTALL.md` 口径），`bash install.sh`。
2. 安装完成后立刻验证三件事：
   - `python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"` → 期望 torch `2.3.1+cu121`、numpy `1.26.4`；
   - `python -c "import pytorch3d"`；
   - `python -c "import mmpose, mmcv; print(mmcv.__version__)"`。
3. `bash fetch_data.sh` 需要交互输入 4 组账号（SMPL / SMPL-X / AGORA / CameraHMR）。**先确认授权是否可申请**，这一步是外部依赖，可能直接卡死整个计划。
4. 产出目录与 B 隔离：建议 `outputs_physic/<run_id>/`，不要写进 `ground-workspace/`。

### P1 可行性实测（只跑 1 张图，只回答三个数）

| 编号 | 要测什么 | 通过标准 | 备注 |
|---|---|---|---|
| P1-1 | 12 GB + offload 下能否完整跑完 1 张图 | 能产出 `humanscene.ply` 且非空 | 跑不通则整个路线终止，转「借 40 GB 卡」方案 |
| P1-2 | 单图端到端墙钟时间（含/不含模型加载分别记） | 记录真实值，不设阈值 | 决定是否只能夜间批处理 |
| P1-3 | 峰值显存（`nvidia-smi` 采样）与是否触发 OOM | 记录真实值 | 与 qwen3-vl 共存时另测 |

补充：由于 `install.sh` 用 torch 2.3.1 而 `WSL_SETUP.md` 用 2.5.1，**两者必须分环境**，不能在同一 venv 里互相「借包」。

### P2 与 B 的对照（只有 P1 全通过才做）

1. 选 5–10 张已有 B 基线的图（覆盖瘦身方案 §七 那些 `rmse_m` / `tilt` / `shift` 已记录的样本）。
2. 同一批图分别跑 A 与 B，比较**可对齐的量**：
   - 地面法线夹角（A 的 `plane_normal` vs B 的 `plane_normal`）；
   - 人体最低点到地面的距离（A 的 mesh 最低点 vs B 的 `lowest_after_m`）；
   - 尺度量纲是否一致（不做 `scale` 强制对齐，直接看原始值差异）。
3. **不做**的事：不把 A 的 `scale` 拿来「修正」B 的深度；不做 A 与 B 的绝对误差排名（A 无 ground truth 指标口径）。
4. 产出一张对照表，只用于回答「A 在哪些图上明显更合理」，不用于发布数字。

### P3 若确认要走生产（仅记录，不在本次范围）

- 需要自建：服务封装、产物转换（PLY/SMPL-X → 目标骨骼）、GPU 队列、评测脚本；
- 需要外部条件：≥40 GB 卡、商业可用的替代授权模型（SMPL/SMPL-X 学术授权不可直接商用）。

---

## 9. 风险表

| 编号 | 风险 | 影响 | 缓解 |
|---|---|---|---|
| R1 | 12 GB 跑不通 FLUX OmniEraser | 路线 A 在本机直接终止 | P1-1 先验证；失败则改为「借 40 GB 卡离线跑」 |
| R2 | SMPL / SMPL-X / AGORA / CameraHMR 授权拿不到 | 数据缺失，Stage 2/3 不可用 | 提前走授权申请；未获批前不排期 |
| R3 | pytorch3d 源码编译失败 / mmcv 1.3.9 与 torch 2.3.1 不兼容 | 环境搭不起来 | 独立 env + 逐步验证（P0 第 2 步）；不要动 B 的环境 |
| R4 | offload 下与 qwen3-vl 抢显存 | 两个服务都变慢或 OOM | A 只在夜间/独占时段跑，不与服务并发 |
| R5 | 无 Evaluation code，无法证明精度收益 | 「换 A 更准」无法被验收 | 明确 A 只做定性参考；要数字必须先自建评测（P2） |
| R6 | 交付链路引入学术授权依赖 | 合规风险 | A 不进交付链路，仅内部使用 |
| R7 | 依赖版本漂移（numpy / torch / mm 系） | 环境随时可能坏 | 锁死 env 快照；不做「顺手升级」 |
| R8 | 上游声明 `<27 s` 被误读为本机可达 | 排期与承诺失真 | 文档与汇报一律以 P1-2 实测为准 |

---

## 10. 待确认事项（需人工决策）

1. **是否有 40 GB 以上显卡可用**（自有 / 租用 / 借）？没有的话，A 只能停在文档阶段。
2. **SMPL、SMPL-X、AGORA、CameraHMR 四个学术账号是否已有**？申请周期是否可接受？
3. 完整 PhySIC 的用途到底是哪一个：**难例参考 / 数据生产 / 未来产品能力**？用途不同，投入差别很大。
4. 是否需要「接触图」这一产出？这是 A 相对 B 唯一无法替代的能力，若不需要，A 的优先级应继续下降。
5. 是否接受「离线、分钟级、人工验收」的交付节奏？

---

## 11. 明确不做

- 不替换、不改动现有 ground_pipeline 在线服务与 API 契约；
- 不在 12 GB 机器上对完整 PhySIC 承诺任何吞吐或并发指标；
- 不在 `ground-workspace` / `.venv-ground` 里安装完整 PhySIC 的依赖（numpy / torch / mmcv 版本冲突）；
- 不把 SMPL/SMPL-X 学术授权数据引入对外交付链路；
- 不基于完整版 demo 输出宣称对 B 的精度提升（无同口径指标）；
- 不把 MoGe-2 替换、队列实现等既有计划的改动混入本对比的执行范围。

---

## 附录：证据索引

| 事实 | 文件 | 位置 |
|---|---|---|
| 需 ≥40 GB 显存（OmniEraser/FLUX） | `INSTALL.md` | Hardware Requirements |
| conda `python=3.10` + `bash install.sh` + CUDA 12.1 | `INSTALL.md` | CUDA / Conda 段 |
| torch 2.3.1+cu121、pytorch3d 源码编译、mmcv 1.3.9、收尾 `numpy==1.26.4` | `install.sh` | 全文 |
| 8 模型加载、`>60 GB` 才关 offload、逐图产出 pkl + ply | `run_optimizer.py` | L1–L60 |
| 三段优化与损失权重 | `cfg/v1.yaml` | `opt_1` / `opt_2` / `opt_3` |
| 4 个学术账号与下载清单 | `fetch_data.sh` | 交互段 |
| Evaluation code 未发布 | `README.md` | Roadmap |
| OmniEraser = FLUX.1-dev、LangSAM large、固定 1024² + 种子 42 | `image_helpers.py` | `load_gsam` / `load_omni` / `get_inpainted_image_omni` |
| 三阶段主干与 `scale` 可优化 | `optimizer.py` | L94–L436 |
| B 的模型与端点 | `ground_service.py` / `GROUND_SERVICE_API.md` | L2308/L2339；§3 |
| B 的冷启动 2 min、单次 10–30 s、超时建议 120 s | `GROUND_SERVICE_API.md` | L30 / L643 |
| B 的显存 7.8 GB、基线指标、单人/平坦地面假设 | `PhySIC 瘦身方案…md` | §七 / §八 / §九 |
