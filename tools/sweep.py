"""同じ入力・同じ時刻で 1 軸だけ振った比較画像を作る。

なぜ要るか:

意匠は目で決めるものだが、「目で決めた」と「なんとなく触った」は外から区別が
つかない。1 軸だけを振った並びが 1 枚あれば、どの値をなぜ選んだかが後から
説明できる。逆に、並べてみて差が読めない軸は触る価値が無かったということ。

決定性について:

絵の側は決定的にできる (app.py は --frames 指定時に uTime をフレーム番号から
作る)。**深度は決定的にならない**。推論は非同期で、描画を待たせない設計なので、
実行ごとに更新回数が変わり正規化 EMA の状態がわずかにずれる。静止画を入力に
すれば収束するので、残差は平均 0.38/255 程度。1 軸を振った差はこれより
はるかに大きいので比較は成立するが、**画素単位の一致は期待しないこと。**

使い方:

    uv run python tools/sweep.py --source shots/20260730-212438_raw.png --param chroma
    uv run python tools/sweep.py --source data/clip.mov --all -- --oklab
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# 教科書 §6.2 の順序。土台から順に決め、split から触り始めない
AXES: dict[str, list[float]] = {
    "chroma": [0.70, 0.90, 1.10, 1.30],
    "inject": [0.00, 0.14, 0.28, 0.42],
    "split":  [0.00, 0.30, 0.60, 0.90],
    "memory": [0.00, 0.30, 0.60, 0.90],
}

OUT_DIR = Path("docs/images")


def _run_one(source: str, param: str, value: float, frames: int,
             passthrough: list[str], tmp: Path) -> tuple[np.ndarray, dict]:
    shot = tmp / f"{param}_{value:.3f}.png"
    cmd = [
        sys.executable, "-m", "ombrelle.app",
        "--source", source, "--frames", str(frames),
        "--shot", str(shot), f"--{param}", str(value),
        "--palette", "--no-hud", "--wait-ready",
        *passthrough,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not shot.exists():
        raise RuntimeError(f"{param}={value} の実行に失敗しました:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    meta_path = shot.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return cv2.imread(str(shot)), meta


def _label(img: np.ndarray, lines: list[str]) -> np.ndarray:
    out = img.copy()
    h = 24 * len(lines) + 14
    cv2.rectangle(out, (0, 0), (330, h), (18, 20, 26), -1)
    for i, s in enumerate(lines):
        cv2.putText(out, s, (12, 24 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (238, 240, 246), 2, cv2.LINE_AA)
    return out


def sweep(source: str, param: str, values: list[float], frames: int,
          passthrough: list[str], scale: float) -> Path:
    tiles, records = [], []
    for v in values:
        img, meta = _run_one(source, param, v, frames, passthrough, Path(tempfile.gettempdir()))
        pal = meta.get("palette_out") or {}
        lines = [f"{param} = {v:g}"]
        if pal:
            lines.append(f"R_C {pal['R_C']:.2f}  dL {pal['dL']:+.3f}")
        tiles.append(_label(img, lines))
        records.append({"value": v, **meta})
        print(f"  {param}={v:<6g} " + (f"R_C {pal['R_C']:.3f}" if pal else "(人物なし)"), flush=True)

    strip = cv2.hconcat(tiles)
    if scale != 1.0:
        strip = cv2.resize(strip, (int(strip.shape[1] * scale), int(strip.shape[0] * scale)),
                           interpolation=cv2.INTER_AREA)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"sweep_{param}.png"
    cv2.imwrite(str(out), strip)
    out.with_suffix(".json").write_text(
        json.dumps({"source": source, "param": param, "frames": frames,
                    "passthrough": passthrough, "runs": records}, indent=2) + "\n",
        encoding="utf-8")
    print(f"→ {out}  ({strip.shape[1]}x{strip.shape[0]})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="1 軸だけ振った比較画像を作る")
    ap.add_argument("--source", required=True, help="静止画 / 動画 / synthetic")
    ap.add_argument("--param", choices=sorted(AXES), help="振る軸")
    ap.add_argument("--values", type=float, nargs="+", help="既定の刻みを上書きする")
    ap.add_argument("--all", action="store_true", help="教科書 §6.2 の順に 4 軸すべて")
    ap.add_argument("--frames", type=int, default=400,
                    help="1 枚あたりのフレーム数。深度と正規化が落ち着くまで要る")
    ap.add_argument("--scale", type=float, default=0.5, help="出力の縮小率")
    ap.add_argument("rest", nargs="*",
                    help="`--` の後ろは app.py へそのまま渡す (--oklab --depth student など)")
    args = ap.parse_args()

    if not args.all and not args.param:
        ap.error("--param か --all のどちらかを指定してください")

    passthrough = [a for a in args.rest if a != "--"]
    params = list(AXES) if args.all else [args.param]
    for p in params:
        vals = args.values if (args.values and not args.all) else AXES[p]
        print(f"[{p}] {vals}")
        sweep(args.source, p, vals, args.frames, passthrough, args.scale)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
