"""深度の正規化。

これは今日いちばん重要な実装上の要点。

単眼深度推定が返すのは**相対**逆深度で、絶対スケールを持たない。フレームごとに
素の min-max 正規化をかけると、画面内にたまたま近い物が入った/出ただけで
マップ全体のスケールが跳ね、絵の側では

  * 筆のサイズが一斉に変わる
  * 彩度と霞の量が一斉に変わる

という形で「絵全体が呼吸する」フリッカになる。深度の**精度**の問題ではなく、
**時間的一貫性**の問題。だから min/max そのものを EMA で追従させる。

  * 外れ値に引かれないようパーセンタイルを使う (min→2%, max→98%)
  * 追従は非対称にする: レンジが広がる方向には素早く、縮む方向にはゆっくり。
    人が急にカメラに近づいたとき、絵が付いてこないと違和感になるが、
    人がフレームから出たときにレンジが急に縮むと画面全体が跳ねる。
"""

from __future__ import annotations

import numpy as np


class DepthNormalizer:
    def __init__(
        self,
        lo_pct: float = 2.0,
        hi_pct: float = 98.0,
        expand: float = 0.35,
        shrink: float = 0.04,
        min_span: float = 1e-3,
    ) -> None:
        self.lo_pct = lo_pct
        self.hi_pct = hi_pct
        self.expand = expand   # レンジが広がる向きの追従率(速い)
        self.shrink = shrink   # レンジが縮む向きの追従率(遅い)
        self.min_span = min_span
        self.lo: float | None = None
        self.hi: float | None = None

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        """raw: 相対逆深度(大きいほど近い) → 0=遠 1=近 に正規化した float32"""
        lo = float(np.percentile(raw, self.lo_pct))
        hi = float(np.percentile(raw, self.hi_pct))
        if self.lo is None or self.hi is None:
            self.lo, self.hi = lo, hi
        else:
            self.lo += (self.expand if lo < self.lo else self.shrink) * (lo - self.lo)
            self.hi += (self.expand if hi > self.hi else self.shrink) * (hi - self.hi)
        span = max(self.hi - self.lo, self.min_span)
        out = (raw - self.lo) / span
        return np.clip(out, 0.0, 1.0).astype(np.float32)


def robust_unit(raw: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0) -> np.ndarray:
    """1 枚単位の頑健正規化。0=遠 1=近。

    蒸留の学習目標に使う。時間 EMA を掛けないのは、学習は 1 枚ずつ独立に扱うため。
    実行時の時間的安定性は DepthNormalizer が別に担保する。
    """
    lo = float(np.percentile(raw, lo_pct))
    hi = float(np.percentile(raw, hi_pct))
    span = max(hi - lo, 1e-6)
    return np.clip((raw - lo) / span, 0.0, 1.0).astype(np.float32)


def temporal_consistency(prev: np.ndarray, cur: np.ndarray) -> float:
    """連続フレーム間の平均絶対差。絵の破綻に直結する指標。

    静止したシーンでこの値が小さいほど、絵は落ち着いて見える。
    AbsRel や δ1 が同じでもこの値が違えば、体験としては別物になる。
    """
    return float(np.abs(cur.astype(np.float32) - prev.astype(np.float32)).mean())
