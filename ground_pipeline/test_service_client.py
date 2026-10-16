"""临时测试客户端：调用 ground_service 的 /v1/ground 接口。"""
import json
import sys
import urllib.request

base = "http://127.0.0.1:%s" % (sys.argv[2] if len(sys.argv) > 2 else "8001")
image = sys.argv[1] if len(sys.argv) > 1 else "/mnt/g/UGit/Phy-SIC/images/man_exercise2.jpg"

req = json.dumps({"image": image}).encode()
r = urllib.request.Request(f"{base}/v1/ground", data=req, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(r, timeout=300) as resp:
    d = json.loads(resp.read().decode())

keep = ["ok", "plane_normal", "plane_d", "camera_height_m", "focal_px", "depth_range_m",
        "image_hw", "floor_pixels", "plane_inliers", "plane_points", "inlier_points"]
out = {k: (len(d[k]) if isinstance(d[k], list) else d[k]) for k in keep}
print(json.dumps(out, ensure_ascii=False))
