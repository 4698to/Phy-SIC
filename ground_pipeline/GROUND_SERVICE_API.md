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
| `distance_threshold` | number | 否 | `0.02` | 平面拟合距离阈值（米），控制哪些点算"属于地面平面" |
| `iterations` | integer | 否 | `256` | RANSAC 迭代次数 |
| `grid_steps` | integer | 否 | `100` | 平面网格分辨率（返回 `grid_steps × grid_steps` 个平面采样点） |
| `visualize` | bool | 否 | `false` | 为 `true` 时返回可视化图片（见下方"可视化输出"） |

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
| `plane_points` | `[[x,y,z], ...]` | 地面上网格采样点（`grid_steps²` 个，相机坐标系，米） |
| `inlier_points` | `[[x,y,z], ...]` | 地面内点 3D 坐标采样（最多 2000 个，随机抽样，相机坐标系，米） |
| `visualization` | object \| null | 仅 `visualize=true` 时返回，见下方"可视化输出" |

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
  "visualization": {
    "overlay": "data:image/png;base64,iVBORw0KGgo...",
    "cloud": "data:image/png;base64,iVBORw0KGgo..."
  }
}
```

#### 可视化输出（`visualize: true`）

响应中的 `visualization` 为两个 PNG data URI（可直接用于 `<img src="...">`）：

| key | 内容 | 用途 |
|---|---|---|
| `overlay` | 原图叠加半透明色块：绿色 = LangSAM 分割的地面 mask，蓝色 = RANSAC 平面内点 | 直观检查分割与拟合是否对准地面 |
| `cloud` | matplotlib 3D 点云图：内点按深度着色，橙色网格为拟合的地面平面 | 检查平面拟合质量、相机高度与地面朝向 |

> 注意：`visualize` 会增加约 0.5~1s 与 ~300KB 响应体积，仅在需要人工确认时开启。

#### 坐标系约定

- **相机坐标系**：原点在相机光心，`x` 向右，`y` 向下，`z` 向前（标准反投影 `x=(u-cx)·z/fx`，`y=(v-cy)·z/fy`），单位为**米**。
- `plane_normal` 主分量被固定为正，因此**朝向可能指向地面也可能反向**；若需要朝上/朝下的朝向，请结合相机位姿或 `plane_d` 的符号自行归一化。
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
for key in ("overlay", "cloud"):
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
