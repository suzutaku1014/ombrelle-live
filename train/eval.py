"""teacher vs student の評価。

3 種類の数字を出す。3 つ目が今回のいちばんの主張。

  1. 精度      … 教師を基準とした MAE / 許容誤差内率 / AbsRel / RMSE
  2. 速度      … 推論のみのレイテンシと FPS、パラメータ数
  3. 時間的一貫性 … 連続フレーム間の深度差

3 が要る理由: この作品にとって深度は「絵のパラメータ」なので、1 枚の精度より
フレーム間の落ち着きの方が体験を支配する。同じ MAE でも、時間的にガタつく方は
筆サイズと彩度が一斉に揺れて絵が呼吸してしまう。数値の良し悪しと絵の破綻は
一致しない、という点を測って示す。

使い方:
    uv run python -m train.eval --data data/room --ckpt checkpoints/student.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ombrelle.depth import StudentDepth, TeacherDepth
from ombrelle.normalize import DepthNormalizer, robust_unit, temporal_consistency
from ombrelle.source import open_source
from train.distill import DepthPairs, evaluate
from train.student import StudentNet


def bench_paired(models: dict, frame: np.ndarray, n: int = 40) -> dict:
    """複数モデルを**交互に**測る。

    連続して片方を n 回測ると、GPU のクロックや熱の状態が測定中に変わり、
    先に測った方が有利/不利になる。実測で teacher の forward は同じ入力でも
    16〜20ms の幅で揺れた。A/B を交互に回してペアで比較すればドリフトは相殺される。
    """
    for m in models.values():
        m.infer(frame)
    ts = {k: [] for k in models}
    for _ in range(n):
        for k, m in models.items():
            t = time.perf_counter()
            m.infer(frame)
            ts[k].append(time.perf_counter() - t)
    out = {}
    for k, v in ts.items():
        a = np.array(v) * 1000.0
        out[k] = {"median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
                  "min": float(a.min()), "std": float(a.std())}
    return out


def consistency(model, source: str, frames: int = 90, use_ema: bool = True) -> float:
    """連続フレームに対する平均絶対差。小さいほど絵が落ち着く。"""
    src = open_source(source, 1280, 720)
    src.wait_first()
    norm = DepthNormalizer()
    prev = None
    diffs = []
    last_seq = -1
    got = 0
    try:
        while got < frames:
            f, _, seq = src.latest()
            if f is None or seq == last_seq:
                time.sleep(0.002)
                continue
            last_seq = seq
            raw = model.infer(f)
            d = norm(raw) if use_ema else robust_unit(raw)
            if prev is not None:
                diffs.append(temporal_consistency(prev, d))
            prev = d
            got += 1
    finally:
        src.stop()
    return float(np.mean(diffs)) if diffs else 0.0


def comparison_image(teacher, student, frame: np.ndarray, out: Path) -> Path:
    """カメラ / 教師 / 生徒 / 差分 を 1 枚に並べる。"""
    dt = robust_unit(teacher.infer(frame))
    ds = robust_unit(student.infer(frame))
    h, w = dt.shape
    ds = cv2.resize(ds, (w, h), interpolation=cv2.INTER_LINEAR)
    cam = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), (w, h))

    def tint(d: np.ndarray) -> np.ndarray:
        img = np.clip(d, 0, 1)[..., None]
        far = np.array([0.22, 0.10, 0.07], np.float32)
        near = np.array([0.70, 0.93, 1.00], np.float32)
        return ((far + (near - far) * img) * 255).astype(np.uint8)

    diff = np.abs(dt - ds)
    dmax = max(float(diff.max()), 1e-3)
    dvis = cv2.applyColorMap((np.clip(diff / dmax, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    tiles = [cam, tint(dt), tint(ds), dvis]
    labels = ["input", "teacher", "student (ours)", f"|diff| max={dmax:.3f}"]
    for t, s in zip(tiles, labels):
        cv2.putText(t, s, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="teacher vs student の評価")
    ap.add_argument("--data", default="data/room", help="精度評価に使う val split")
    ap.add_argument("--ckpt", default="checkpoints/student.pt")
    ap.add_argument("--source", default="synthetic", help="時間的一貫性の測定に使う入力")
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--out", default="docs/bench.md")
    ap.add_argument("--figure", default="docs/depth_compare.png")
    args = ap.parse_args()

    teacher = TeacherDepth()
    student = StudentDepth(args.ckpt)
    device = student.device

    # ---- 1. 精度 (教師を基準とした val split)
    acc = {}
    if Path(args.data, "labels.npy").exists():
        va = DepthPairs(args.data, "val", augment=False)
        dl = DataLoader(va, batch_size=16, shuffle=False, num_workers=2)
        blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        net = StudentNet(width=blob.get("width", 1.0))
        net.load_state_dict(blob["state_dict"])
        net.to(device).eval()
        acc = evaluate(net, dl, device)
        print("accuracy vs teacher:", json.dumps(acc, indent=2))

    # ---- 2. 速度
    src = open_source(args.source, 1280, 720)
    frame = src.wait_first()
    src.stop()
    bm = bench_paired({"teacher": teacher, "student": student}, frame)
    t_med, t_p90 = bm["teacher"]["median"], bm["teacher"]["p90"]
    s_med, s_p90 = bm["student"]["median"], bm["student"]["p90"]
    print("latency:", json.dumps(bm, indent=2))

    # ---- 3. 時間的一貫性 (EMA 正規化の有無も測る)
    tc = {
        "teacher_ema": consistency(teacher, args.source, args.frames, True),
        "teacher_raw": consistency(teacher, args.source, args.frames, False),
        "student_ema": consistency(student, args.source, args.frames, True),
        "student_raw": consistency(student, args.source, args.frames, False),
    }
    print("temporal consistency:", json.dumps(tc, indent=2))

    fig = comparison_image(teacher, student, frame, Path(args.figure))

    rows = [
        "# ベンチマーク\n",
        f"入力 `{args.source}` / 精度評価 `{args.data}` の val split / device `{device}`\n",
        "## 速度とサイズ\n",
        "前処理 + forward + 読み戻しを含む1フレームの推論時間。teacher と student を",
        "交互に測ってGPUの熱ドリフトを相殺している。\n",
        "| モデル | パラメータ | median | p90 | min | std | FPS(推論のみ) |",
        "|---|---|---|---|---|---|---|",
        f"| teacher (Depth Anything V2 Small) | {teacher.params/1e6:.2f} M | "
        f"{t_med:.1f} ms | {t_p90:.1f} ms | {bm['teacher']['min']:.1f} ms | "
        f"{bm['teacher']['std']:.1f} | {1000/t_med:.0f} |",
        f"| **student (自作蒸留)** | **{student.params/1e6:.2f} M** | "
        f"**{s_med:.1f} ms** | **{s_p90:.1f} ms** | {bm['student']['min']:.1f} ms | "
        f"{bm['student']['std']:.1f} | **{1000/s_med:.0f}** |",
        f"\nパラメータ **{teacher.params/student.params:.1f}x 削減**、"
        f"推論 **{t_med/s_med:.2f}x 高速化**。",
        "パラメータ比ほど速くならないのは、この規模だとカーネル起動と",
        "メモリ帯域が支配的で FLOPs が律速ではないため。\n",
    ]
    if acc:
        rows += [
            "## 精度 (教師を基準にした相対深度、正規化 [0,1])\n",
            "| 指標 | student |",
            "|---|---|",
            f"| MAE | {acc['mae']:.4f} |",
            f"| 誤差 < 0.05 の画素率 | {acc['acc<0.05']:.3f} |",
            f"| 誤差 < 0.10 の画素率 | {acc['acc<0.10']:.3f} |",
            f"| AbsRel (gt>0.05 でマスク) | {acc['absrel']:.4f} |",
            f"| RMSE | {acc['rmse']:.4f} |",
            "\nAbsRel は正規化した目標値に対しては gt≈0 (最遠) で発散するため、"
            "gt>0.05 でマスクして算出している。\n",
        ]
    rows += [
        "## 時間的一貫性 (連続フレーム間の平均絶対差、小さいほど絵が落ち着く)\n",
        "| モデル | 毎フレーム min-max | 非対称 EMA 正規化 |",
        "|---|---|---|",
        f"| teacher | {tc['teacher_raw']:.4f} | {tc['teacher_ema']:.4f} |",
        f"| student | {tc['student_raw']:.4f} | {tc['student_ema']:.4f} |",
        "\nこの表が本題。深度は絵のパラメータ(筆サイズ/彩度/霞)を駆動するので、"
        "1 枚の精度よりフレーム間の落ち着きが体験を支配する。"
        "毎フレーム min-max 正規化と非対称 EMA を比べると、"
        f"teacher で {tc['teacher_raw']/max(tc['teacher_ema'],1e-9):.1f}x、"
        f"student で {tc['student_raw']/max(tc['student_ema'],1e-9):.1f}x の差が出る。\n",
        f"![depth comparison]({Path(args.figure).name})\n",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows), encoding="utf-8")
    print(f"wrote {out} and {fig}")


if __name__ == "__main__":
    main()
