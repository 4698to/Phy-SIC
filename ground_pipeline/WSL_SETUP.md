# WSL 对接指南：ground_pipeline 单图地面提取

本文记录在 WSL2/Linux 环境中，把 RGB 图片接到 `ground_pipeline` 的准备步骤。当前仓库已有地面几何拟合核心，但还没有端到端的模型推理命令；分割和深度模型必须先生成 mask、metric depth 与相机内参，再调用几何流水线。

## 1. 流程与输入输出

```text
RGB 图片
  ├─ LangSAM：GroundingDINO 文本检测 + SAM 2.1 Hiera Small 分割 → floor_mask (H×W)
  ├─ DepthPro → metric depth (H×W，米) + focal length (像素)
  ├─ focal length + 图片宽高 → K (3×3)
  └─ ground_pipeline.extract_ground(RGB, depth, K, floor_mask) → plane 参数及几何数据
```

核心接口在 `ground_pipeline/pipeline.py`。深度、mask、RGB 必须有相同 H×W；深度需为米制正值；K 是 3×3 相机内参矩阵。`extract_ground` 用地面候选点进行 RANSAC/SVD 平面拟合，并返回 `plane_normal`、`plane_d`、`plane_points`、`plane_inlier_mask` 等数组。CLI `python -m ground_pipeline` 目前只接收已经推理好的深度、mask 和 K，不会自行加载模型。

## 2. WSL 路径

确保 Windows 磁盘已挂载。此项目的 Windows 路径对应：

```bash
cd /mnt/g/UGit/Phy-SIC
IMAGE='/mnt/e/视频.library/images/LZQV5MH4UP4YG.info/5306c1d3gy1hsktgsqco6j20sg11wq4s.jpg'
```

路径必须加引号，避免中文路径或特殊字符被 shell 拆开。若 WSL 没有 `/mnt/g` 或 `/mnt/e`，先检查 `/etc/wsl.conf` 的 automount 设置，或将代码和图片复制到 WSL 的 Linux 文件系统。

## 3. 建立隔离环境

不要在此阶段直接改动 GVHMR Windows 环境。示例使用 Conda 和 Python 3.10：

```bash
conda create -n ground-wsl python=3.10 -y
conda activate ground-wsl
cd /mnt/g/UGit/Phy-SIC
```

WSL 中的 NVIDIA GPU 需要 Windows 主机安装支持 WSL 的 NVIDIA 驱动。先验证：

```bash
nvidia-smi
```

安装 PyTorch 时按 PyTorch 官方安装页面选择与驱动匹配的 Linux/CUDA wheel。可供起点的组合是 PyTorch 2.5.1 + torchvision 0.20.1 + CUDA 12.1 wheel；它满足 SAM 2 官方当前安装要求以及仓库 LangSAM 对 PyTorch 的最低声明。若改用仓库 pin 的旧 SAM 2 版本，仍建议先用此组合并按下面顺序逐项验证。不要安装 CUDA Toolkit，除非确实要编译 SAM 2 自定义扩展。

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")'
```

如果最后显示 CUDA 不可用，应先解决 WSL GPU 驱动/设备透传，再继续，不要误以为模型权重会自动解决该问题。

## 4. Python 包安装

LangSAM 在仓库内，SAM 2 版本在它的 `requirements.txt` 中固定；安装这个 pin 可以避免直接跟随上游最新版造成代码 API 不匹配。此处安装 LangSAM 时使用 `--no-deps`，避免其完整服务依赖（如 Gradio、LitServe）和解析器覆盖已选的 Torch 版本；单图推理再补充必需包。

```bash
python -m pip install \
  'transformers>=4.44.2,<5' \
  hydra-core iopath addict yapf \
  opencv-python-headless pillow-heif
python -m pip install \
  'sam-2 @ git+https://github.com/facebookresearch/segment-anything-2@c2ec8e14a185632b0a5d8b161928ceb50197eddc'
python -m pip install -e ./external/lang-segment-anything --no-deps
python -m pip install -e ./external/ml-depth-pro --no-deps
python -m pip install -e ./ground_pipeline --no-deps
```

DepthPro 项目依赖 `torch`、`torchvision`、`timm`、`numpy<2`、`pillow_heif` 等；LangSAM 还依赖 Transformers、SAM 2、Hydra/OmegaConf。若 pip 报依赖缺失，先确认缺哪个包，再只补该包，不要运行仓库根目录的完整安装脚本（它包含与本任务无关的模型和 PyTorch3D）。

在 Linux 上安装 SAM 2 时可能编译可选 CUDA 扩展；上游说明扩展编译失败时多数图像推理仍可工作，但可能缺少部分后处理功能。WSL 下优先照 SAM 2 上游的当前安装说明处理编译问题，不要盲目设置 CUDA 扩展环境变量；编译需要的 nvcc 版本必须与 PyTorch CUDA 版本一致。

## 5. 模型权重与缓存位置

建议统一把显式权重放进项目的数据目录，并忽略这些大文件，不要提交到 Git：

| 模型/缓存 | 下载或来源 | 目标位置 |
|---|---|---|
| DepthPro 权重 | https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt | `/mnt/g/UGit/Phy-SIC/data/depth_pro.pt`（Windows 对应 `G:\\UGit\\Phy-SIC\\data\\depth_pro.pt`）。这是仓库 DepthPro 默认配置的相对路径，推理时从项目根目录启动。 |
| SAM 2.1 Hiera Small | https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt | `/mnt/g/UGit/Phy-SIC/data/sam2.1_hiera_small.pt`。LangSAM 初始化时传入该 checkpoint 路径。 |
| GroundingDINO | Hugging Face 模型 ID `IDEA-Research/grounding-dino-base`：https://huggingface.co/IDEA-Research/grounding-dino-base | Transformers 首次加载时自动缓存，默认在 `~/.cache/huggingface/hub`。也可设置 `HF_HOME` 改到有空间的位置。 |

DepthPro 权重约 1.9 GB；首次下载前确认磁盘空间与网络可达。SAM checkpoint 使用 PyTorch Hub 下载时会进入 Torch cache；显式传入 checkpoint 路径可避免依赖隐式缓存位置。模型权重受各自上游许可证约束。

## 6. 单图模型对接参考

仓库当前没有封装该端到端步骤。以下是建议的 Python 对接顺序（在项目根目录运行，完成安装和权重下载后）。先通过 LangSAM 获取 floor mask，释放分割模型显存后再运行 DepthPro；这适合显存有限的 GPU：

```python
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lang_sam import LangSAM
import depth_pro
from ground_pipeline import extract_ground, save_result

image_path = Path('/mnt/e/视频.library/images/LZQV5MH4UP4YG.info/5306c1d3gy1hsktgsqco6j20sg11wq4s.jpg')
image_path = image_path.expanduser()

# 以 DepthPro 的读取/EXIF 朝向处理作为统一 RGB 坐标，避免模型输入错位。
rgb, _, exif_f_px = depth_pro.load_rgb(image_path)
image_pil = Image.fromarray(rgb)

# GroundingDINO 文本提示 + SAM 生成一个或多个候选地面 mask。
segmenter = LangSAM(sam_type='sam2.1_hiera_small', ckpt_path='data/sam2.1_hiera_small.pt')
segmentation = segmenter.predict([image_pil], ['ground, floor, carpet.'])[0]
masks = np.asarray(segmentation['masks'], dtype=bool)
if masks.ndim == 2:
    floor_mask = masks
elif masks.ndim == 3 and masks.shape[0] > 0:
    floor_mask = masks.any(axis=0)
else:
    raise RuntimeError('No floor mask returned; inspect GroundingDINO detections/prompt.')
del segmenter
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# DepthPro 输出米制深度和像素焦距。其默认权重路径为 ./data/depth_pro.pt。
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, transform = depth_pro.create_model_and_transforms(device=device)
model.eval()
image_tensor = transform(rgb)
with torch.inference_mode():
    prediction = model.infer(image_tensor, f_px=exif_f_px)
depth = prediction['depth'].squeeze().detach().cpu().numpy()
f_px = float(prediction['focallength_px'].detach().cpu().item())
del model, transform, image_tensor, prediction
if torch.cuda.is_available():
    torch.cuda.empty_cache()

height, width = rgb.shape[:2]
K = np.array([[f_px, 0, width / 2], [0, f_px, height / 2], [0, 0, 1]], dtype=np.float32)
result = extract_ground(rgb, depth, K, floor_mask)
output = Path('ground_pipeline/outputs/5306c1d3_ground.npz')
output.parent.mkdir(parents=True, exist_ok=True)
save_result(result, output)
print(f'Saved: {output}; inliers={result["plane_inlier_mask"].sum()}; n={result["plane_normal"]}; d={result["plane_d"]}')
```

> 该代码是对接草案，不是当前仓库现成 CLI。具体运行前应先在 WSL 小样本验证 LangSAM 返回的 `masks` 形状、DepthPro 输出的深度尺寸和 focal 字段类型；如模型 API 有版本差异，以实际安装版本为准。地面提示可能返回多个实例，合并前应目视检查 mask，避免把墙面、地毯边缘或其他水平物体一并当作地面。

## 7. 几何 CLI（已有预计算输入时）

若已另行生成并检查 `depth.npy`（米）、`floor_mask.png`（白色为地面）和 `K.npy`，可以只运行现有 CLI：

```bash
python -m ground_pipeline \
  --image "$IMAGE" \
  --depth /path/to/depth.npy \
  --mask /path/to/floor_mask.png \
  --intrinsics /path/to/K.npy \
  --output ground_pipeline/outputs/5306c1d3_ground.npz
```

NPZ 可用 NumPy 检查：

```python
import numpy as np
with np.load('ground_pipeline/outputs/5306c1d3_ground.npz') as data:
    print(data.files)
    print('plane_normal:', data['plane_normal'])
    print('plane_d:', data['plane_d'])
    print('inliers:', int(data['plane_inlier_mask'].sum()))
```

## 8. 排错检查

1. `torch.cuda.is_available()` 为 False：检查 Windows NVIDIA 驱动、WSL 版本和 `nvidia-smi`，不要在 WSL 内安装 Windows CUDA 驱动。
2. SAM2 编译失败：确认安装的 SAM 2 代码与仓库 pin 一致；上游说明可选 CUDA 扩展失败时多数推理仍能运行。若 Python 包或配置找不到，先确认 SAM2 仓库已正确安装。
3. GroundingDINO 首次加载失败：检查 Hugging Face 网络/缓存空间；可设置 `HF_HOME=/path/to/cache` 后重试。
4. `extract_ground` 报尺寸不匹配：确保 RGB、mask、depth 都采用同一个 EXIF 旋转与 resize 结果。深度图不得只缩放而不对应调整相机内参。
5. 有效地面点少于 3 个或平面拟合失败：先检查 floor mask、深度单位/无效值与分割结果；必要时人工修正 mask，而非增大 RANSAC 次数掩盖输入错误。
6. 不要把 `data/*.pt`、Hugging Face cache、生成的 `ground_pipeline/outputs/` 或输入样本加入仓库提交。

## 上游资料

- [SAM 2 安装与 checkpoint 说明](https://github.com/facebookresearch/sam2)
- [Apple DepthPro README 与推理接口](https://github.com/apple/ml-depth-pro)
- [LangSAM 源码与依赖](https://github.com/luca-medeiros/lang-segment-anything)
- [GroundingDINO Transformers 模型](https://huggingface.co/IDEA-Research/grounding-dino-base)
