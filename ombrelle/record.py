"""描いた結果が「どこで」「どれだけ」動いているかを測る。

なぜ要るか:

静止画のスクショでは時間の話が原理的に見えない。「筆が回り続ける」「ちらつく」
という指摘に対して 1 枚の絵をいくら見比べても検証できず、実際それで 2 回
見当違いの直し方をした (静止時の微風 / 深度の値のブレ)。

動画を残す手もあるが、圧縮が時間方向のノイズを平滑化してしまうので、
**測りたいものが記録の過程で消える**。ここでは非圧縮のまま差分だけを積算し、

  * 画面のどこが動いているかの地図 (PNG 1 枚)
  * 動きの量の数値

を出す。地図を見れば「顔だけ」「平らな壁だけ」「全面均一」の区別がつき、
どの経路が原因かが 1 回で分かる。
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


class VideoRecorder:
    """描いた結果をそのまま動画に落とす。

    画面収録との違いは 3 つ。ウィンドウの枠も HUD も入らない (FBO を読むため)、
    内部解像度そのままで記録される、そして**再生速度が実時間と一致する**。

    最後の点が地味に効く。記録中は読み戻しと符号化で fps が落ちるので、
    描画のフレームをそのまま並べると早回しになる。目標 fps に足りる分だけ
    間引いて書く。
    """

    def __init__(self, path: str | Path, size: tuple[int, int], fps: float = 30.0) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self._w = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self._w.isOpened():
            raise RuntimeError(f"動画を開けませんでした: {self.path}")
        self._next = 0.0
        self.frames = 0
        self.started: float | None = None

    def add(self, rgb: np.ndarray, t: float) -> None:
        if self.started is None:
            self.started, self._next = t, t
        if t + 1e-6 < self._next:
            return
        self._next += 1.0 / self.fps
        self._w.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self.frames += 1

    def close(self, t: float) -> dict:
        self._w.release()
        dur = max(t - (self.started or t), 1e-6)
        return {"path": str(self.path), "frames": self.frames,
                "seconds": round(dur, 2), "fps": round(self.frames / dur, 1)}


class MotionProbe:
    """連続フレームを読み戻して時間方向の変化を積算する。

    録っている間は FBO の読み戻しで GPU を止めるので fps が落ちる。
    計測の間だけ動かすこと。
    """

    def __init__(self, frames: int = 90, scale: float = 0.5) -> None:
        self.frames = frames
        self.scale = scale
        self.active = False
        self._prev: np.ndarray | None = None
        self._acc: np.ndarray | None = None      # 画素ごとの |差| の和
        self._n = 0

    def start(self) -> None:
        self.active = True
        self._prev = None
        self._acc = None
        self._n = 0

    def add(self, rgb: np.ndarray) -> bool:
        """1 フレーム取り込む。必要な枚数に達したら True。"""
        if not self.active:
            return False
        img = rgb
        if self.scale != 1.0:
            img = cv2.resize(rgb, (int(rgb.shape[1] * self.scale), int(rgb.shape[0] * self.scale)),
                             interpolation=cv2.INTER_AREA)
        cur = img.astype(np.float32)
        if self._prev is not None:
            d = np.abs(cur - self._prev).mean(axis=2)
            self._acc = d if self._acc is None else self._acc + d
            self._n += 1
        self._prev = cur
        if self._n >= self.frames:
            self.active = False
            return True
        return False

    def report(self, path: str | Path, meta: dict | None = None) -> dict:
        """地図を PNG で、数値を JSON で書き出す。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._acc is None or self._n == 0:
            raise RuntimeError("フレームが足りません")

        m = self._acc / self._n                      # 画素ごとの毎フレーム平均変化 (0..255)
        stats = {
            "frames": self._n,
            "mean": round(float(m.mean()), 4),
            "p50": round(float(np.percentile(m, 50)), 4),
            "p99": round(float(np.percentile(m, 99)), 4),
            "max": round(float(m.max()), 4),
            # 粒子感のディザは全画素に一様に乗る (±0.012*255 ≒ 3)。
            # それを超えて動いている画素の割合が、実際に「動いて見える」量
            "over_3": round(float((m > 3.0).mean()), 4),
            "over_6": round(float((m > 6.0).mean()), 4),
        }
        if meta:
            stats.update(meta)

        # 地図の目盛りは**絶対値**にする。p99 で正規化すると、静止していても
        # 画面いっぱいが真っ赤になって「動いていない」ことが読めない。
        # 下限を 4 に固定すると、粒子感だけの状態は暗いまま、実際に動いている
        # 場所だけが明るくなる
        hi = max(float(np.percentile(m, 99)), 4.0)
        heat = cv2.applyColorMap(np.clip(m / hi * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        stats["scale"] = round(hi, 2)
        cv2.putText(heat, f"motion mean {stats['mean']:.2f}  p99 {stats['p99']:.2f}  "
                          f">3: {stats['over_3']*100:.1f}%   (scale 0-{hi:.1f})",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(path), heat)
        path.with_suffix(".json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return stats
