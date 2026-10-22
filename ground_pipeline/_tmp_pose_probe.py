import base64
import json
import sys
import urllib.request
from pathlib import Path

IMAGE = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/g/UGit/Phy-SIC/images/yoga_core.jpg")
PREFIX = Path(sys.argv[2] if len(sys.argv) > 2 else "yoga_core").name
OUT_DIR = Path("/mnt/g/UGit/Phy-SIC/images")
URL = "http://127.0.0.1:8004/v1/ground"

payload = {
    "image": "data:image/jpeg;base64," + base64.b64encode(IMAGE.read_bytes()).decode(),
    "pose_inference": True,
    "pose_want_mesh": True,
    "pose_want_overlay": True,
    "visualize": True,
}
request = urllib.request.Request(
    URL,
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=900) as resp:
    body = json.loads(resp.read().decode())

pose = body.get("pose") or {}
print("pose.status:", pose.get("status"))
print("pose.mesh_error:", pose.get("mesh_error"))
print("pose.alignment:", json.dumps(pose.get("alignment"), ensure_ascii=False, indent=2))

viz = body.get("visualization") or {}
for key, filename in [
    ("overlay", f"{PREFIX}_overlay.png"),
    ("views", f"{PREFIX}_views.png"),
    ("pose_overlay", f"{PREFIX}_pose_overlay.png"),
    ("pose_mesh", f"{PREFIX}_pose_mesh.png"),
    ("pose_world", f"{PREFIX}_pose_world_3view.png"),
]:
    data = viz.get(key)
    if not data:
        print(key, "MISSING")
        continue
    target = OUT_DIR / filename
    target.write_bytes(base64.b64decode(data.split(",", 1)[1]))
    print(key, "->", target)
