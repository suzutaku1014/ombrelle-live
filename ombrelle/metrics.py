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


def palette_line(label: str, stats) -> str:
    """人物と背景の関係を1行で出す。

    R_C が目標帯 (palette.TARGET_RC) を外れているときだけ `*` を付ける。
    「今どちらへ外れているか」が一目で分かることが目的なので、
    帯の中では飾りを付けない。
    """
    if stats is None:
        return f"palette {label:<4} --"
    mark = "*" if not stats.in_target() else " "
    return (f"palette {label:<4} dL {stats.dL:+6.3f}  R_C {stats.Rc:5.2f}{mark} "
            f"dh {stats.dh:4.2f}  area {stats.area:4.2f}")


def hud_lines(meter: Meter, state, energy: float, depth_source: str,
              stats_in=None, stats_out=None, stab=None, raw_energy: float = 0.0) -> list[str]:
    # 入力の比と出力の比を並べる。「入力 4.07 倍 → 圧縮後 1.65 倍」のように、
    # 処理系が何をしたかは差でしか読めない
    if getattr(state, "palette", False):
        palette = [palette_line("in", stats_in), palette_line("out", stats_out)]
        if getattr(state, "stabilize", False) and stab is not None:
            palette.append(f"stabilize ON   人物の彩度 x{stab.subj_chroma:4.2f}   "
                           f"分割 x{stab.split_scale:4.2f}")
    else:
        palette = ["palette  off  (a で計測  x で自動補正)"]
    return [
        f"view {int(state.view)}:{_VIEW_NAMES.get(int(state.view), '?')}   "
        f"{meter.fps:5.1f} fps   e2e {meter.latency_ms:5.1f} ms",
        f"depth {depth_source:<8} {meter.ms('depth'):5.1f}ms   "
        f"seg {meter.ms('seg'):4.1f}ms   flow {meter.ms('flow'):4.1f}ms",
        f"compose {'ON  stand ' + format(state.stand, '4.2f') if state.compose > 0.5 else 'OFF'}",
        # raw は不感帯を掛ける前。静止時にこれを読んで dead を決める
        f"energy {energy:6.4f} (raw {raw_energy:6.4f})   dead {state.flow_dead:4.2f}   "
        f"flowGain {state.flow_gain:4.1f}   lod {state.cam_lod:3.1f}   "
        f"camEMA {state.cam_ema:4.2f}   idleWind {state.idle_wind:4.2f}",
        f"haze {state.haze:4.2f}  chroma {state.chroma:4.2f}  brush {state.brush:4.2f}  "
        f"split {state.split:4.2f}   色空間 {'Oklab' if getattr(state, 'oklab', 0.0) > 0.5 else 'luma'}",
        *palette,
        "v b brush  t y split  f g inject  i o memory  k l haze  n m chroma  , . lod  w e wind",
        "c compose  r u stand  0-3 view  s shot  p save  d depth  a palette  j oklab",
        "z cam-ema   ; ' flow-dead   x stabilize   h hud   q quit",
    ]
