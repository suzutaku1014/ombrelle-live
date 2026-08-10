"""人物と背景の色の関係を測る。

なぜ要るか:

現在の彩度制御 (`uChroma` → `compand`) は**画面全体に一律のスカラー**を掛ける。
ところが「白い壁で顔だけ派手」という実写の失敗は、絶対彩度ではなく
**領域間の比**の問題だった (入力の顔/壁の彩度比 4.07 倍 → 圧縮後 1.65 倍)。
比を見ていない限り、場面ごとの当たり外れは説明できない。

ここでは3つだけ測る。

  dL  = L(人物) - L(背景)        明度差。形と読みやすさを支配する
  R_C = C(人物) / C(背景)        彩度比。「浮く / 埋もれる」を支配する
  dh  = 円周距離(h人物, h背景)   色相差。焦点と調和のトレードオフ

背景として画面全体ではなく**人物の外周リング**を使う。同時対比は局所現象で、
画面の反対側にある色は人物の見えにほとんど効かないため。

測るだけで何も制御しない。制御 (background_stability) は次の段階で、
まず「どの場面でどの値になるか」を実測してから目標帯を決める。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .oklab import hue_distance, rgb8_to_oklab

# 教科書の A/B 開始値。普遍的な定数ではなく、このプロジェクトの実験の出発点。
TARGET_RC = (1.4, 1.8)


@dataclass(frozen=True)
class RegionStats:
    dL: float
    Rc: float
    dh: float
    L_subj: float
    C_subj: float
    L_surr: float
    C_surr: float
    area: float          # 人物が画面に占める割合

    def in_target(self) -> bool:
        return TARGET_RC[0] <= self.Rc <= TARGET_RC[1]

    def as_dict(self) -> dict:
        return {
            "dL": round(self.dL, 4),
            "R_C": round(self.Rc, 3),
            "dh": round(self.dh, 3),
            "L_subj": round(self.L_subj, 4),
            "C_subj": round(self.C_subj, 4),
            "L_surr": round(self.L_surr, 4),
            "C_surr": round(self.C_surr, 4),
            "area": round(self.area, 4),
        }


class PaletteMeter:
    """人物マットから領域統計を作る。CPU 側で完結する (GPU は描画に残す)。"""

    def __init__(
        self,
        size: tuple[int, int] = (256, 144),
        thresh: float = 0.6,
        ring_out: int = 15,
        subj_in: int = 5,
        min_area: float = 0.02,
        ema: float = 0.85,
    ) -> None:
        self.size = size
        self.thresh = thresh
        self.ring_out = ring_out
        self.subj_in = subj_in
        self.min_area = min_area
        self.ema = ema
        # EMA は導出値ではなく素の量に掛ける。比や角度を直接平滑化すると
        # 分母が小さいフレームで跳ね、色相は 2π の巻き戻りで壊れる
        self._s: dict[str, np.ndarray] | None = None
        self.stats: RegionStats | None = None

    def reset(self) -> None:
        self._s = None
        self.stats = None

    def update(self, rgb: np.ndarray, matte: np.ndarray) -> RegionStats | None:
        """rgb: uint8 (H,W,3) / matte: float (h,w) 0..1 → 平滑化済みの統計。

        人物が小さすぎる (既定 2% 未満) 場合は None を返し、状態も捨てる。
        """
        w, h = self.size
        small = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        m = cv2.resize(matte.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        mask = (m > self.thresh).astype(np.uint8)

        area = float(mask.mean())
        if area < self.min_area:
            self.reset()
            return None

        # 人物側はマットの縁を避けて内側だけを見る。境界付近は背景の色が
        # 混ざっており、そこを人物の色として数えると比が必ず 1 に寄る
        k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.subj_in, self.subj_in))
        subj = cv2.erode(mask, k_in)
        if subj.sum() < 32:          # 細い人物では内側が消える。その時は素のマット
            subj = mask

        # 背景側は人物に**隣接する**帯だけ。画面の反対側の色は見えに効かない
        k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.ring_out, self.ring_out))
        ring = (cv2.dilate(mask, k_out) > 0) & (mask == 0)
        if ring.sum() < 32:          # 人物が画面を覆っている。比較対象が無い
            self.reset()
            return None

        lab = rgb8_to_oklab(small)
        raw = {
            "subj": _region(lab, subj.astype(bool)),
            "surr": _region(lab, ring),
        }

        if self._s is None:
            self._s = {k: v.copy() for k, v in raw.items()}
        else:
            for k, v in raw.items():
                self._s[k] += (1.0 - self.ema) * (v - self._s[k])

        s = self._s
        L_s, C_s, a_s, b_s = s["subj"]
        L_r, C_r, a_r, b_r = s["surr"]
        self.stats = RegionStats(
            dL=float(L_s - L_r),
            Rc=float(C_s / max(C_r, 1e-4)),
            dh=hue_distance(float(np.arctan2(b_s, a_s)), float(np.arctan2(b_r, a_r))),
            L_subj=float(L_s), C_subj=float(C_s),
            L_surr=float(L_r), C_surr=float(C_r),
            area=area,
        )
        return self.stats


def _region(lab: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """領域の代表値 (L, C, a, b) を返す。

    L と C は**中央値**。少数の白飛び画素に引かれないため。
    色相は中央値を取れない (円周上の量で、-π と +π が隣接する)。
    代わりに (a, b) ベクトルの平均を取り、後で atan2 する。
    彩度の高い画素ほど自然に重みが大きくなるので、無彩色領域の
    ノイズだらけの色相に引きずられない。
    """
    px = lab[mask]
    C = np.hypot(px[:, 1], px[:, 2])
    return np.array([
        float(np.median(px[:, 0])),
        float(np.median(C)),
        float(px[:, 1].mean()),
        float(px[:, 2].mean()),
    ], dtype=np.float64)
