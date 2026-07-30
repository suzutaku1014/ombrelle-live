"""蒸留用データの収集。

教師 (Depth Anything V2) に疑似ラベルを付けさせるので、人手のアノテーションは要らない。
ただし「カメラを回した分だけデータが増える」わけではない点に注意が必要:

  * 60fps で 1 分回すと 3600 枚だが、隣接フレームはほぼ同一画像。
    そのまま学習すると実質的なデータ量は数十枚しかない。
    → **前に採用したフレームとの差分が閾値未満なら捨てる**(近重複の除去)
  * ラベルは 1 枚単位の頑健正規化 [0,1]。レンダラが消費する量に直接合わせる
  * 保存はフレームを jpg、ラベルを 1 本の .npy に積む。ラベルは入力の 1/2 解像度
    (student の出力解像度に合わせる)

使い方:
    uv run python -m train.collect --source cam:0 --seconds 240 --out data/room
    uv run python -m train.collect --source synthetic --frames 1200 --out data/synth
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from ombrelle.depth import TEACHER_SIZE, TeacherDepth, resize_uint8
from ombrelle.normalize import robust_unit
from ombrelle.source import open_source


def main() -> None:
    ap = argparse.ArgumentParser(description="教師の疑似ラベル付きデータを収集する")
    ap.add_argument("--source", default="cam:0")
    ap.add_argument("--out", default="data/room")
    ap.add_argument("--frames", type=int, default=0, help="採用フレーム数の上限")
    ap.add_argument("--seconds", type=float, default=0.0, help="収集時間の上限")
    ap.add_argument("--novelty", type=float, default=3.0,
                    help="採用条件: 前の採用フレームとの平均絶対差(0-255)がこの値以上")
    ap.add_argument("--cam-width", type=int, default=1280)
    ap.add_argument("--cam-height", type=int, default=720)
    args = ap.parse_args()
    if not args.frames and not args.seconds:
        args.seconds = 120.0

    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)

    src = open_source(args.source, args.cam_width, args.cam_height)
    src.wait_first()
    teacher = TeacherDepth()
    tw, th = TEACHER_SIZE
    lw, lh = tw // 2, th // 2

    labels: list[np.ndarray] = []
    prev_small: np.ndarray | None = None
    last_seq = -1
    seen = skipped = 0
    t0 = time.perf_counter()
    print(f"collecting → {out}  (novelty>={args.novelty})")

    try:
        while True:
            if args.frames and len(labels) >= args.frames:
                break
            if args.seconds and time.perf_counter() - t0 > args.seconds:
                break
            frame, stamp, seq = src.latest()
            if frame is None or seq == last_seq:
                time.sleep(0.002)
                continue
            last_seq = seq
            seen += 1

            # 推論時とまったく同じ resize 経路を通す(学習/推論のドメイン一致)
            img = resize_uint8(frame, (tw, th), teacher.device)
            small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (64, 36),
                               interpolation=cv2.INTER_AREA).astype(np.float32)
            if prev_small is not None and np.abs(small - prev_small).mean() < args.novelty:
                skipped += 1
                continue
            prev_small = small

            raw = teacher.infer(frame)
            lab = robust_unit(raw)
            lab = cv2.resize(lab, (lw, lh), interpolation=cv2.INTER_AREA)

            idx = len(labels)
            cv2.imwrite(str(out / "frames" / f"{idx:06d}.jpg"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            labels.append(lab.astype(np.float16))
            if idx % 100 == 0:
                el = time.perf_counter() - t0
                print(f"  kept {idx:5d} / seen {seen:6d} (skip {skipped}) "
                      f"{el:5.1f}s", flush=True)
    finally:
        src.stop()

    if not labels:
        raise SystemExit("採用フレームが 0 枚でした。--novelty を下げてください")

    np.save(out / "labels.npy", np.stack(labels))
    meta = {
        "source": args.source,
        "input_size": [tw, th],
        "label_size": [lw, lh],
        "count": len(labels),
        "seen": seen,
        "skipped_near_duplicate": skipped,
        "novelty_threshold": args.novelty,
        "teacher": "depth-anything/Depth-Anything-V2-Small-hf",
        "seconds": round(time.perf_counter() - t0, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
