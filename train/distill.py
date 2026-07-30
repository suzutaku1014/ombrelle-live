"""教師 → 生徒の蒸留学習。

ML の衛生管理として重要な点:

  * **train/val はランダム分割してはいけない。** 動画から採ったフレームは時間的に
    隣接するものが近重複なので、ランダム分割すると val に train のほぼ同じ画像が
    漏れ、スコアが実力より良く出る。連続したブロックで切る。
  * 幾何変換の増強は左右反転だけにする。ラベルは「画像全体で頑健正規化した値」なので、
    ランダムクロップするとその正規化が壊れる(切り取った領域の最遠点は元の最遠点ではない)。
    測光系の増強(明度/コントラスト/彩度/ノイズ)は自由に掛けられる。

使い方:
    uv run python -m train.distill --data data/room --epochs 30
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
from torch.utils.data import DataLoader, Dataset

from ombrelle.depth import IMAGENET_MEAN, IMAGENET_STD, pick_device
from train.student import StudentNet, distill_loss


class DepthPairs(Dataset):
    def __init__(self, root: str | Path, split: str = "train", val_frac: float = 0.15,
                 augment: bool = True, blocks: int = 8) -> None:
        root = Path(root)
        self.root = root
        self.labels = np.load(root / "labels.npy")  # (N, lh, lw) float16
        meta = json.loads((root / "meta.json").read_text())
        self.input_size = tuple(meta["input_size"])
        n = len(self.labels)

        # 連続ブロック単位で分割する(近重複の漏れを防ぐ)
        edges = np.linspace(0, n, blocks + 1).astype(int)
        val_idx, train_idx = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            cut = b - int((b - a) * val_frac)
            train_idx.extend(range(a, cut))
            val_idx.extend(range(cut, b))
        self.idx = np.array(train_idx if split == "train" else val_idx)
        self.augment = augment and split == "train"

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        j = int(self.idx[i])
        bgr = cv2.imread(str(self.root / "frames" / f"{j:06d}.jpg"), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        lab = self.labels[j].astype(np.float32)

        if self.augment:
            if np.random.rand() < 0.5:
                img = img[:, ::-1].copy()
                lab = lab[:, ::-1].copy()
            # 測光系のみ: 照明条件へのロバスト性を狙う
            img = img * (0.75 + 0.5 * np.random.rand())                # 明度
            m = float(img.mean())
            img = m + (img - m) * (0.75 + 0.5 * np.random.rand())      # コントラスト
            g = img.mean(axis=2, keepdims=True)
            img = g + (img - g) * (0.6 + 0.8 * np.random.rand())       # 彩度
            img = img + np.random.randn(*img.shape).astype(np.float32) * (0.02 * np.random.rand())
            img = np.clip(img, 0.0, 1.0)

        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)), dtype=np.float32))
        y = torch.from_numpy(np.ascontiguousarray(lab, dtype=np.float32))[None]
        return x, y


def evaluate(model, loader, device) -> dict:
    model.eval()
    mae = d05 = d10 = absrel = rmse = 0.0
    n = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x)
            if p.shape[-2:] != y.shape[-2:]:
                p = F.interpolate(p, size=y.shape[-2:], mode="bilinear", align_corners=False)
            p = p.clamp(0.0, 1.0)
            e = (p - y).abs()
            b = x.shape[0]
            mae += e.mean().item() * b
            d05 += (e < 0.05).float().mean().item() * b
            d10 += (e < 0.10).float().mean().item() * b
            # 正規化された値に対する AbsRel は gt≈0 で発散するのでマスクする
            m = y > 0.05
            absrel += (e[m] / y[m]).mean().item() * b if m.any() else 0.0
            rmse += ((p - y) ** 2).mean().sqrt().item() * b
            n += b
    model.train()
    return {"mae": mae / n, "acc<0.05": d05 / n, "acc<0.10": d10 / n,
            "absrel": absrel / n, "rmse": rmse / n}


def main() -> None:
    ap = argparse.ArgumentParser(description="深度推定の蒸留学習")
    ap.add_argument("--data", nargs="+", default=["data/room"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="checkpoints/student.pt")
    ap.add_argument("--run", default="runs/distill")
    args = ap.parse_args()

    device = pick_device()
    tr = torch.utils.data.ConcatDataset([DepthPairs(d, "train") for d in args.data])
    va = torch.utils.data.ConcatDataset([DepthPairs(d, "val", augment=False) for d in args.data])
    input_size = DepthPairs(args.data[0], "val", augment=False).input_size
    print(f"train {len(tr)}  val {len(va)}  device {device}  input {input_size}")

    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                       drop_last=True, persistent_workers=args.workers > 0)
    dl_va = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    model = StudentNet(width=args.width).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"student {params / 1e6:.2f} M params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * max(1, len(dl_tr)), pct_start=0.25)

    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    log = (run / "log.jsonl").open("w")
    best = float("inf")
    t0 = time.perf_counter()

    for ep in range(1, args.epochs + 1):
        model.train()
        acc = {"loss": 0.0, "l1": 0.0, "grad": 0.0}
        nb = 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            p = model(x)
            if p.shape[-2:] != y.shape[-2:]:
                p = F.interpolate(p, size=y.shape[-2:], mode="bilinear", align_corners=False)
            out = distill_loss(p, y)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            for k in acc:
                acc[k] += float(out[k].detach())
            nb += 1
        for k in acc:
            acc[k] /= max(nb, 1)

        m = evaluate(model, dl_va, device)
        rec = {"epoch": ep, "sec": round(time.perf_counter() - t0, 1),
               "lr": sched.get_last_lr()[0], **acc, **m}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"ep{ep:3d} loss {acc['loss']:.4f}  val mae {m['mae']:.4f}  "
              f"acc<0.05 {m['acc<0.05']:.3f}  absrel {m['absrel']:.4f}  "
              f"({rec['sec']:.0f}s)", flush=True)

        if m["mae"] < best:
            best = m["mae"]
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "size": list(input_size),
                "width": args.width,
                "params": params,
                "metrics": m,
                "epoch": ep,
                "data": args.data,
            }, args.out)

    log.close()
    print(f"best val mae {best:.4f} → {args.out}")


if __name__ == "__main__":
    main()
