"""意匠パラメータのスイープ。

1 枚の画像に対して複数のパラメータ組を描き、1 枚のコンタクトシートにまとめる。

絵は目で決めるしかないが、1 パラメータずつ実行し直して記憶で比べるのは効率が悪く、
判断もぶれる。**同じ画・同じ瞬間で並べて比べる**ための道具。

    uv run python -m tools.sweep --source shots/xxx_raw.png --grid brush=1.8,3.0 split=0.2,0.5
    uv run python -m tools.sweep --source shots/xxx_raw.png --preset overview
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import cv2
import numpy as np

from ombrelle.color import WhiteBalance
from ombrelle.flow import FlowField
from ombrelle.gl.renderer import Renderer
from ombrelle.source import open_source

DEFAULTS = {
    "brush": 2.4, "split": 0.50, "haze": 0.35, "chroma": 1.35,
    "cam_lod": 2.0, "flow_gain": 1.5, "paint_mix": 1.0,
}

PRESETS = {
    "overview": {"brush": [1.2, 2.4, 3.6], "split": [0.0, 0.35, 0.7]},
    "brush":    {"brush": [0.8, 1.4, 2.0, 2.8, 3.6, 5.0]},
    "color":    {"chroma": [1.0, 1.3, 1.6], "split": [0.2, 0.45, 0.7]},
    "haze":     {"haze": [0.0, 0.3, 0.6, 0.9]},
    "lod":      {"cam_lod": [0.5, 1.5, 2.5, 3.5]},
}


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (18, 20, 26), -1)
    cv2.putText(out, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (240, 242, 248), 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="意匠パラメータのスイープ")
    ap.add_argument("--source", required=True)
    ap.add_argument("--depth", choices=["teacher", "student", "off"], default="teacher")
    ap.add_argument("--student-ckpt", default="checkpoints/student.pt")
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None)
    ap.add_argument("--grid", nargs="*", default=[], help="例: brush=1.8,3.0 split=0.2,0.5")
    ap.add_argument("--fixed", nargs="*", default=[], help="例: haze=0.3 chroma=1.4")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--out", default="shots/sweep.png")
    ap.add_argument("--tiles", default="", help="個別タイルも保存するディレクトリ")
    args = ap.parse_args()

    axes: dict[str, list[float]] = {}
    if args.preset:
        axes.update({k: list(v) for k, v in PRESETS[args.preset].items()})
    for spec in args.grid:
        k, v = spec.split("=", 1)
        axes[k] = [float(x) for x in v.split(",")]
    if not axes:
        raise SystemExit("--preset か --grid を指定してください")

    base = dict(DEFAULTS)
    for spec in args.fixed:
        k, v = spec.split("=", 1)
        base[k] = float(v)

    src = open_source(args.source, 1280, 720, mirror=False)
    frame = src.wait_first()

    renderer = Renderer(win_w=args.width, win_h=args.height,
                        render_w=args.width, render_h=args.height,
                        title="ombrelle sweep")
    renderer.update_camera(frame)
    wb = WhiteBalance(ema=0.0)
    wb.update(frame)
    print(f"white balance gain {wb.gain}")

    flowf = FlowField()
    flowf.update(frame, 0.0)
    flowf.update(frame, 1 / 60)   # 静止画なのでフローは 0。既定の風だけが残る

    depth_arr = None
    if args.depth != "off":
        from ombrelle.depth import build
        from ombrelle.normalize import DepthNormalizer
        model = build(args.depth, args.student_ckpt)
        depth_arr = DepthNormalizer()(model.infer(frame))
        renderer.update_depth(depth_arr)
        print(f"depth[{args.depth}] {depth_arr.shape}")

    def seed_depth(xy):
        if depth_arr is None:
            return 0.6
        h, w = depth_arr.shape[:2]
        return float(depth_arr[int((1 - xy[1]) * (h - 1)), int(xy[0] * (w - 1))])

    keys = list(axes)
    combos = list(itertools.product(*(axes[k] for k in keys)))
    print(f"{len(combos)} 通り: " + " x ".join(f"{k}{axes[k]}" for k in keys))

    tiles = []
    tile_dir = Path(args.tiles) if args.tiles else None
    if tile_dir:
        tile_dir.mkdir(parents=True, exist_ok=True)

    for combo in combos:
        cfg = dict(base)
        cfg.update(dict(zip(keys, combo)))
        renderer.draw({
            "uTime": 8.0, "uAdv": 4.0, "uView": 0.0,
            "uFlowGain": cfg["flow_gain"], "uCamLod": cfg["cam_lod"],
            "uSeed": (float(flowf.centroid[0]), float(flowf.centroid[1])),
            "uSeedDepth": seed_depth(flowf.centroid),
            "uWind": (float(flowf.wind[0]), float(flowf.wind[1])),
            "uEnergy": 0.0, "uPaintMix": cfg["paint_mix"],
            "uHaze": cfg["haze"], "uChroma": cfg["chroma"],
            "uBrush": cfg["brush"], "uSplit": cfg["split"],
            "uWhite": tuple(float(x) for x in wb.gain),
        })
        raw = renderer.fbo.read(components=3, alignment=1)
        img = np.flipud(np.frombuffer(raw, np.uint8).reshape(args.height, args.width, 3))
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        name = " ".join(f"{k}={cfg[k]:g}" for k in keys)
        if tile_dir:
            cv2.imwrite(str(tile_dir / (name.replace(" ", "_").replace("=", "") + ".png")), bgr)
        tiles.append(label(bgr, name))

    cols = min(args.cols, len(tiles))
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    if len(rows[-1]) < cols:
        pad = np.zeros_like(tiles[0])
        rows[-1] += [pad] * (cols - len(rows[-1]))
    sheet = np.vstack([np.hstack(r) for r in rows])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(f"wrote {out}  ({sheet.shape[1]}x{sheet.shape[0]})")

    src.stop()
    renderer.close()


if __name__ == "__main__":
    main()
