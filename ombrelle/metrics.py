"""計測と HUD。

end-to-end レイテンシ = 「カメラが撮った瞬間」から「その絵を画面に出した瞬間」まで。
段ごとの処理時間の合計ではなく、実際に体験される遅延を測る。
"""

from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np


class Meter:
    def __init__(self, window: int = 90) -> None:
        self.frame_dt = deque(maxlen=window)
        self.latency = deque(maxlen=window)
        self.stage: dict[str, deque] = {}
        self._last = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        self.frame_dt.append(dt)
        return dt

    def add_latency(self, seconds: float) -> None:
        self.latency.append(seconds)

    def add_stage(self, name: str, seconds: float) -> None:
        self.stage.setdefault(name, deque(maxlen=60)).append(seconds)

    @property
    def fps(self) -> float:
        if not self.frame_dt:
            return 0.0
        m = sum(self.frame_dt) / len(self.frame_dt)
        return 1.0 / m if m > 0 else 0.0

    def ms(self, name: str) -> float:
        d = self.stage.get(name)
        if not d:
            return 0.0
        return 1000.0 * sum(d) / len(d)

    @property
    def latency_ms(self) -> float:
        if not self.latency:
            return 0.0
        return 1000.0 * sum(self.latency) / len(self.latency)


_VIEW_NAMES = {0: "paint", 1: "camera", 2: "depth", 3: "flow"}


def build_hud(width: int, height: int, lines: list[str]) -> np.ndarray:
    """左上に半透明の板と文字を描いた RGBA 画像を返す(OpenCV 座標系 = y 下向き)。"""
    hud = np.zeros((height, width, 4), dtype=np.uint8)
    pad, lh = 12, 20
    box_h = pad * 2 + lh * len(lines)
    box_w = 320
    cv2.rectangle(hud, (0, 0), (box_w, box_h), (12, 14, 20, 150), -1)
    for i, text in enumerate(lines):
        cv2.putText(
            hud, text, (pad, pad + lh * (i + 1) - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (235, 238, 245, 255), 1, cv2.LINE_AA,
        )
    return hud


def hud_lines(meter: Meter, state, energy: float, depth_source: str) -> list[str]:
    return [
        f"view {int(state.view)}:{_VIEW_NAMES.get(int(state.view), '?')}   "
        f"{meter.fps:5.1f} fps   e2e {meter.latency_ms:5.1f} ms",
        f"depth {depth_source:<8} {meter.ms('depth'):5.1f}ms   "
        f"seg {meter.ms('seg'):4.1f}ms   flow {meter.ms('flow'):4.1f}ms",
        f"compose {'ON  stand ' + format(state.stand, '4.2f') if state.compose > 0.5 else 'OFF'}",
        f"energy {energy:6.4f}   flowGain {state.flow_gain:4.1f}   lod {state.cam_lod:3.1f}",
        f"haze {state.haze:4.2f}  chroma {state.chroma:4.2f}  brush {state.brush:4.2f}  split {state.split:4.2f}",
        "v b brush   t y split   f g inject   k l haze   n m chroma   , . lod",
        "c compose   r u stand   0-3 view   s shot   p save   d depth   q quit",
    ]
