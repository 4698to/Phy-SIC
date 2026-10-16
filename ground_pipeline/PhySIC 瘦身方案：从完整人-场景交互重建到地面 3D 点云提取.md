# PhySIC 瘦身方案：从完整人-场景交互重建到地面 3D 点云提取

## 一、目标

保留 PhySIC 单图、固定相机、可见地面场景下的核心能力，仅提取**地面 3D 点云**，移除人体重建与联合物理优化，将管线从多模型融合优化压缩为三步轻量流程。

## 二、PhySIC 原始流程回顾

PhySIC 的完整流程分为三个阶段：

| 阶段 | 内容 | 涉及模型 |
|---|---|---|
| Stage 1 | 场景重建 | SAM2、OmniEraser、DepthPro、MoGe |
| Stage 2 | 人体重建与对齐 | CameraHMR、WiLoR、SMPL-X |
| Stage 3 | 联合物理优化 | 置信度加权优化、接触先验、穿透约束 |

Stage 1 中已经包含地面平面拟合能力：SAM2 分割地面掩码 → 3D 点云分割 → RANSAC 拟合平面 → 补充采样地面点。这恰好是你需要的核心。

## 三、瘦身原则

1. **只保留地面相关分支**：移除 Stage 2 和 Stage 3，仅保留 Stage 1 中“地面掩码 → 反投影 → 平面拟合”这条路径。
2. **合并同类模型**：用单个轻量语义分割模型替代 SAM2 的多轮分割，用单个轻量深度模型替代 DepthPro + MoGe 双模型融合。
3. **去掉非必要环节**：OmniEraser 背景修复、尺度/平移 RANSAC 融合、补充采样地面点均可省略。
4. **精度换轻量**：接受地面掩码不完整、深度图有噪声，通过 RANSAC 内点阈值过滤来保证鲁棒性。

## 四、瘦身后流程

### 4.1 三步管线

**第一步：轻量语义分割，得到人物和地面掩码**

- **推荐模型**：YOLO26n-sem（1.6M 参数，RTX 3090 上 4.4ms）或 LWSegNet（4.1M 参数，Jetson Orin Nano 量化后 8.4ms）。
- **替代关系**：一次前向同时输出 person 和 road/sidewalk 两个类别，替代 PhySIC 中 SAM2 对人物和地面的多轮分割。
- **输出**：人物掩码 + 地面掩码。

**第二步：轻量单目深度估计，得到度量深度图**

- **推荐模型**：MetaDepth-CPU（最小 2.8MB，手机 CPU 上 15-30 FPS）。
- **替代关系**：单模型直接输出度量深度，替代 PhySIC 中 DepthPro + MoGe 的双模型融合及后续尺度/平移 RANSAC 优化。
- **输出**：与 RGB 对齐的度量深度图。

**第三步：地面反投影与平面拟合**

- **操作**：取地面掩码内像素，结合深度值和相机内参逐像素反投影为相机坐标系 3D 点；对 3D 点做 RANSAC 平面拟合，过滤离群点。
- **保留关系**：保留 PhySIC 中 RANSAC 拟合地面平面的核心逻辑，但输入简化为地面掩码反投影点，不需要先融合完整场景点云。
- **输出**：地面平面方程 + 地面 3D 点云。

### 4.2 流程图

flowchart TB
    subgraph PhySIC[PhySIC 原始流程]
        direction TB
        P_A[输入: 单张 RGB 图片] --> P_B[Stage 1: 场景重建]
        P_B --> P_B1[SAM2 分割人体掩码]
        P_B1 --> P_B2[OmniEraser 修复被遮挡背景]
        P_B2 --> P_B3[DepthPro 度量深度 + MoGe 相对几何]
        P_B3 --> P_B4[RANSAC 优化尺度/平移, 融合点云]
        P_B4 --> P_B5[SAM2 分割地面掩码]
        P_B5 --> P_B6[RANSAC 拟合地面平面 + 补充采样]

        P_B6 --> P_C[Stage 2: 人体重建与对齐]
        P_C --> P_C1[CameraHMR + WiLoR 初始化 SMPL-X]
        P_C1 --> P_C2[人体点云与度量场景尺度对齐]

        P_C2 --> P_D[Stage 3: 联合物理优化]
        P_D --> P_D1[置信度加权联合优化]
        P_D1 --> P_D2[精炼姿态/相机/尺度]
        P_D2 --> P_D3[输出: 人体网格 + 场景表面 + 接触图]
    end

    subgraph Lite[轻量化流程]
        direction TB
        L_A[输入: 单张 RGB 图片] --> L_B[轻量语义分割]
        L_A --> L_C[轻量单目深度估计]

        L_B --> L_B1[YOLO26n-sem / LWSegNet]
        L_B1 --> L_B2[输出: 人物掩码 + 地面掩码]

        L_C --> L_C1[MetaDepth-CPU]
        L_C1 --> L_C2[输出: 度量深度图]

        L_B2 --> L_D[取地面掩码区域]
        L_C2 --> L_D
        L_D --> L_E[逐像素反投影 + 相机内参]
        L_E --> L_F[RANSAC 平面拟合 + 过滤离群点]
        L_F --> L_G[输出: 地面 3D 点云]
    end

    P_B2 -.->|替换| L_B1
    P_B3 -.->|替换| L_C1
    P_B6 -.->|保留核心| L_F

    style PhySIC fill:#fce4ec
    style Lite fill:#e8f5e9
    style P_C fill:#fff9c4
    style P_D fill:#fff9c4
    style L_G fill:#c8e6c9

## 五、替换对照表

| PhySIC 环节 | 瘦身方案 | 说明 |
|---|---|---|
| SAM2 分割人体掩码 | YOLO26n-sem / LWSegNet | 一次前向替代多轮分割 |
| SAM2 分割地面掩码 | 同上，取 road/sidewalk 类别 | 无需单独分割流程 |
| OmniEraser 背景修复 | 移除 | 地面掩码缺失部分可接受 |
| DepthPro + MoGe 双模型 | MetaDepth-CPU | 单模型输出度量深度 |
| 尺度/平移 RANSAC 融合 | 移除 | 不需要融合完整场景点云 |
| RANSAC 拟合地面平面 | 保留 | 输入简化为地面掩码反投影点 |
| 补充采样地面点 | 移除 | 精度要求不高，可省略 |
| Stage 2 人体重建与对齐 | 移除 | 不需要人体网格输出 |
| Stage 3 联合物理优化 | 移除 | 不需要接触图与穿透约束 |

## 六、资源对比

| 维度 | PhySIC 原始 | 瘦身方案 |
|---|---|---|
| 模型数量 | 6+（SAM2、OmniEraser、DepthPro、MoGe、CameraHMR、WiLoR 等） | 2（分割 + 深度） |
| 参数量级 | 数十 M 至数百 M | 2-4M |
| 输出 | 人体网格 + 场景表面 + 接触图 | 地面 3D 点云 |
| 优化环节 | 置信度加权联合优化 | 无，仅 RANSAC |
| 处理时间 | 端到端 <27 秒，优化 9 秒 | 预计 <1 秒（取决于硬件） |

## 七、注意事项

1. **RANSAC 鲁棒性**：PhySIC 自述平坦地面假设在 RANSAC 无法找到共识时会失败。瘦身方案中深度模型精度更低，建议设置合理的内点距离阈值来过滤噪声。
2. **地面掩码完整性**：移除 OmniEraser 后，被人体遮挡的地面区域会缺失。如果缺失面积过大影响拟合，可用简单的邻域插值补全，或接受点云中的空洞。
3. **度量尺度**：MetaDepth-CPU 需选择度量输出模式，否则反投影得到的 3D 点云只有相对尺度，无法获得真实物理尺寸。
4. **相机内参**：反投影依赖相机内参。如果内参未知，可先用 GeoCalib 等单图标定工具估计焦距，再代入反投影公式。

## 八、一句话总结

**用“YOLO26n-sem / LWSegNet + MetaDepth-CPU + RANSAC”三步替代 PhySIC 的完整三阶段管线，保留地面平面拟合核心，砍掉人体重建与联合优化，模型参数量从数十 M 降至 2-4M，输出从人体网格 + 场景表面 + 接触图简化为地面 3D 点云。**