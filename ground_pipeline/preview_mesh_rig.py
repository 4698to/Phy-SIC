"""预览 mhr_mesh_fbx 导出的「SAM-3D 网格 + MHR-lod5 骨架」FBX。

直接解析生成后的 ASCII FBX（Vertices / PolygonVertexIndex / 骨 Lcl Translation
与 Connections 层级），输出「原图 + 三视图」PNG，用于目视检查网格与骨架是否贴合。
FBX 本身是 Z-up、厘米，这里坐标直接用厘米画。
"""
from __future__ import annotations

import argparse
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402

# (标签, 水平轴, 垂直轴, 深度轴, 水平轴名, 垂直轴名)
VIEWS = (
    ("FRONT", 0, 2, 1, "X", "Z"),
    ("SIDE", 1, 2, 0, "Y", "Z"),
    ("TOP", 0, 1, 2, "X", "Y"),
)


def _array(text: str, name: str) -> np.ndarray:
    i = text.index("\n\t\t" + name + ": ")
    seg = text[i: text.index("}", i)]
    body = seg[seg.index("a:") + 2:]
    return np.array([float(x) for x in re.findall(r"-?\d+\.?\d*", body)])


def parse_fbx(path: str | pathlib.Path) -> dict:
    """解析 ASCII FBX：网格顶点/面 + 骨层级与累积世界坐标（厘米）。"""
    text = pathlib.Path(path).read_text(encoding="utf-8")
    verts = _array(text, "Vertices").reshape(-1, 3)
    pvi = _array(text, "PolygonVertexIndex").astype(np.int64)
    faces: list[list[int]] = []
    cur: list[int] = []
    for v in pvi:
        if v < 0:
            cur.append(int(-v - 1))
            faces.append(cur)
            cur = []
        else:
            cur.append(int(v))

    models = {int(i): (n, k) for i, n, k in re.findall(
        r'\tModel: (\d+), "Model::([^"]+)", "(Null|LimbNode|Mesh)"', text)}
    loc: dict[int, np.ndarray] = {}
    for mid, (name, kind) in models.items():
        if kind != "LimbNode":
            continue
        i = text.index(f'"Model::{name}", "LimbNode"')
        m = re.search(r'"Lcl Translation", "Lcl Translation", "", "A",'
                      r"([-\d.]+),([-\d.]+),([-\d.]+)", text[i: i + 900])
        loc[mid] = np.array([float(g) for g in m.groups()])
    parent = {int(c): int(p) for c, p in re.findall(r'\tC: "OO",(\d+),(\d+)', text)}

    pos: dict[int, np.ndarray] = {}

    def resolve(mid: int) -> np.ndarray:
        if mid in pos:
            return pos[mid]
        base = loc.get(mid, np.zeros(3))
        p = parent.get(mid, 0)
        pos[mid] = base + (resolve(p) if p else 0.0)
        return pos[mid]

    bones = []
    for mid in loc:
        p = parent.get(mid, 0)
        # 根骨的父节点是 Armature（Null，位于原点），连线会拉出一条横跨全图的长线，
        # 这里只画骨与骨之间的段。
        bones.append({"name": models[mid][0], "pos": resolve(mid), "parent": p,
                      "attach": p in loc,
                      "ppos": resolve(p) if p in loc else np.zeros(3)})
    return {"verts": verts, "faces": np.asarray(faces, dtype=np.int64), "bones": bones}


def draw_view(ax, data: dict, view) -> None:
    label, h, v, d, hx, vx = view
    verts, faces = data["verts"], data["faces"]
    tri = verts[faces]
    depth = tri[:, :, d].mean(axis=1)
    span = float(np.ptp(depth)) or 1.0
    coll = PolyCollection(tri[:, :, [h, v]], array=(depth - depth.min()) / span,
                          cmap="Blues_r", alpha=0.9, edgecolors="none",
                          rasterized=True)
    coll.set_clim(0.0, 1.0)
    ax.add_collection(coll)

    segs = [[b["ppos"][[h, v]], b["pos"][[h, v]]]
            for b in data["bones"] if b["attach"]]
    ax.add_collection(LineCollection(segs, colors="#c1121f", linewidths=1.8, zorder=5))
    pts = np.array([b["pos"][[h, v]] for b in data["bones"]])
    ax.scatter(pts[:, 0], pts[:, 1], s=6, c="#c1121f", zorder=6)

    for b in data["bones"]:
        if not b["attach"]:
            ax.scatter(b["pos"][h], b["pos"][v], s=45, marker="*",
                       c="#ffb703", edgecolors="#8a5a00", linewidths=0.6, zorder=7)

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{label}  ({hx}-> / {vx}^)", fontsize=11)


def render_preview(fbx: str | pathlib.Path, image: str | None,
                   out: pathlib.Path) -> pathlib.Path:
    data = parse_fbx(fbx)
    n = 4 if image else 3
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 5.2))
    axes = list(np.atleast_1d(axes))
    k = 0
    if image:
        try:
            ax_img = axes[k]
            ax_img.imshow(plt.imread(image))
            ax_img.set_title("SOURCE", fontsize=11)
            ax_img.axis("off")
            k += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] 读取原图失败 {image}: {exc}")

    for view in VIEWS:
        draw_view(axes[k], data, view)
        k += 1

    verts = data["verts"]
    h_cm = float(verts[:, 2].max() - verts[:, 2].min())
    fig.suptitle(f"{pathlib.Path(fbx).stem}  |  bones={len(data['bones'])}  "
                 f"verts={len(verts)}  faces={len(data['faces'])}  "
                 f"height={h_cm:.1f} cm", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fbx", nargs="+", help="mesh-rig FBX 文件或目录")
    ap.add_argument("--image-dirs", nargs="*", default=[])
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    files: list[pathlib.Path] = []
    for it in args.fbx:
        p = pathlib.Path(it)
        files.extend(sorted(p.glob("*_mesh_rig.fbx")) if p.is_dir() else [p])
    imgs: dict[str, pathlib.Path] = {}
    for d in args.image_dirs:
        for p in pathlib.Path(d).glob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                imgs.setdefault(p.stem, p)

    for f in files:
        outdir = pathlib.Path(args.outdir) if args.outdir else f.parent
        outdir.mkdir(parents=True, exist_ok=True)
        stem = f.name.replace("_mesh_rig.fbx", "")
        out = render_preview(f, str(imgs[stem]) if stem in imgs else None,
                             outdir / f"{stem}_mesh_rig_preview.png")
        print(f"preview: {out}")


if __name__ == "__main__":
    main()
