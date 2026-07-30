"""オプティカルフロー = 人の動きから作る「風の場」。

設計の要点:
  * 低解像度(320x180)で計算する。人体スケールの動きが欲しいだけで、画素単位の
    精度は不要。むしろ低解像度の方が「大きな流れ」が素直に出る。
  * **空間ブラー + 時間 EMA** を必ずかける。筆触パスは各フラグメントで格子ごと
    回転させるため、角度の場がガタつくと格子が破れて絵が壊れる。滑らかさは
    見た目の好みではなく、レンダラの要求仕様。
  * 単位は「画面を 1 とした **毎秒** の移動量」に正規化して渡す。
    フレームあたりで渡すと、同じ物理的な動きでも 30fps と 120fps で風速が 4 倍違う。
    解像度にもフレームレートにも依存しない量にして初めて、絵の挙動が再現可能になる。
"""

from __future__ import annotations

import time

import cv2
import numpy as np


def gust_env(t: float) -> float:
    """原典 gustEnv() の CPU 版。0.22Hz の呼吸 + 45秒ごとに息を呑む。"""
    br = 0.62 + 0.38 * np.sin(2 * np.pi * 0.22 * t + 0.8 * np.sin(2 * np.pi * 0.031 * t))
    ph = t % 45.0
    if ph > 43.0:
        pf = 1.0 - np.clip((ph - 43.0) / 1.2, 0.0, 1.0)
    else:
        pf = np.clip(ph / 0.9, 0.0, 1.0)
        pf = pf * pf * (3 - 2 * pf)
    return float(max(br * pf, 0.05))


class FlowField:
    def __init__(self, width: int = 320, height: int = 180, ema: float = 0.40) -> None:
        self.width = width
        self.height = height
        self.ema = ema
        self._prev_gray: np.ndarray | None = None
        self._prev_t: float | None = None
        self.field = np.zeros((height, width, 2), dtype=np.float32)
        self.energy = 0.0
        self.centroid = (0.5, 0.5)   # y は上向き(GL 流儀)
        self.wind = (-1.0, 0.12)     # 既定は原典の弧を描く風の向き

    def update(self, rgb: np.ndarray, stamp: float | None = None) -> None:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if stamp is None:
            stamp = time.perf_counter()
        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_t = stamp
            return
        dt = max(stamp - (self._prev_t or stamp), 1e-3)

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=21, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0,
        )
        self._prev_gray = gray
        self._prev_t = stamp

        # 画面を 1 とした「毎秒」の移動量へ
        flow[..., 0] /= float(self.width) * dt
        flow[..., 1] /= float(self.height) * dt
        # 空間を滑らかにする(格子の破れ防止)。テクスチャ自体が 4 倍に拡大されるので
        # ここで掛けすぎると局所的な動きの峰が消える
        flow = cv2.GaussianBlur(flow, (0, 0), sigmaX=3.0, sigmaY=3.0)
        # 時間を滑らかにする
        self.field = self.ema * self.field + (1.0 - self.ema) * flow

        mag = np.linalg.norm(self.field, axis=2)
        self.energy = float(mag.mean()) * 1.0

        total = float(mag.sum())
        if total > 1e-4:
            ys, xs = np.mgrid[0 : self.height, 0 : self.width].astype(np.float32)
            cx = float((xs * mag).sum() / total) / self.width
            cy = float((ys * mag).sum() / total) / self.height
            # 画像 y 下向き → GL y 上向き
            target = (cx, 1.0 - cy)
            # 重心は跳ねやすいので追従を鈍らせる
            self.centroid = (
                0.90 * self.centroid[0] + 0.10 * target[0],
                0.90 * self.centroid[1] + 0.10 * target[1],
            )
            wx = float((self.field[..., 0] * mag).sum() / total)
            wy = float((self.field[..., 1] * mag).sum() / total)
            if abs(wx) + abs(wy) > 1e-6:
                n = (wx * wx + wy * wy) ** 0.5
                # 画像 y 下向き → GL y 上向き
                self.wind = (
                    0.85 * self.wind[0] + 0.15 * (wx / n),
                    0.85 * self.wind[1] + 0.15 * (-wy / n),
                )
