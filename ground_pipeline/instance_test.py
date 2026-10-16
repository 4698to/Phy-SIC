from pathlib import Path
import gc
import sys
import numpy as np
import torch
from PIL import Image
import depth_pro
from lang_sam import LangSAM
from ground_pipeline import extract_ground, save_result

root = Path('/mnt/g/UGit/Phy-SIC')
image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else root / 'images/man_sofa.jpg'
output = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/tmp/man_sofa_ground.npz')
rgb, _, exif_f_px = depth_pro.load_rgb(image_path)
image_pil = Image.fromarray(rgb)
print('image', rgb.shape, 'focal_input', exif_f_px, 'cuda', torch.cuda.is_available(), flush=True)

segmenter = LangSAM(
    sam_type='sam2.1_hiera_small',
    ckpt_path=str(root / 'data/sam2.1_hiera_small_langsam.pt'),
)
segmentation = segmenter.predict([image_pil], ['floor, wooden floor.'])[0]
masks = np.asarray(segmentation['masks'], dtype=bool)
print('detections', len(segmentation['labels']), 'labels', segmentation['labels'], 'mask_shape', masks.shape, flush=True)
if masks.ndim == 2:
    floor_mask = masks
elif masks.ndim == 3 and masks.shape[0] > 0:
    floor_mask = masks.any(axis=0)
else:
    raise RuntimeError('LangSAM returned no floor masks')
del segmenter, segmentation, masks
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, transform = depth_pro.create_model_and_transforms(device=device)
model.eval()
image_tensor = transform(rgb)
with torch.inference_mode():
    prediction = model.infer(image_tensor, f_px=exif_f_px)
depth = prediction['depth'].squeeze().detach().cpu().numpy()
f_px = float(prediction['focallength_px'].detach().cpu().item())
print('depth', depth.shape, 'range', float(np.nanmin(depth)), float(np.nanmax(depth)), 'focal', f_px, 'floor_pixels', int(floor_mask.sum()), flush=True)

del model, transform, image_tensor, prediction
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

height, width = rgb.shape[:2]
intrinsics = np.array([[f_px, 0, width / 2], [0, f_px, height / 2], [0, 0, 1]], dtype=np.float32)
result = extract_ground(rgb, depth, intrinsics, floor_mask)
save_result(result, output)
print('RESULT', output, 'inliers', int(result['plane_inlier_mask'].sum()), 'normal', result['plane_normal'].tolist(), 'd', result['plane_d'], 'plane_points', result['plane_points'].shape, flush=True)
