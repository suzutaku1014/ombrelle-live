"""人物のセグメンテーション。

深度とは**役割が違う**ので別に持つ:

  * 深度 … 奥行きの順序。筆サイズ・空気遠近・花びらのオクルージョンを駆動する
  * マット … 人物の同定。現実とモネ風背景のどちらを描くかを決める

最初は深度の閾値でマットを作ろうとしたが、実写で割れた。人物が明確に手前にある
構図では完璧だったが、部屋の隅で撮った写真では**左の壁も「近い」**ため分離できず、
前景率が 62% になった (本来は 3 割程度)。
「近いもの」と「人」は別の概念で、深度は前者しか知らない。

LRASPP MobileNetV3-Large (torchvision, COCO+VOC) は 3.2M params / 8.3ms で、
同じ 2 枚とも正しく人物だけを取れた。深度 teacher (17.9ms) の半分以下。

境界の精度は要求しない。合成を**筆触パスの前**で行うので、同じ一筆が境界をまたぎ、
マットの粗さは筆に隠れる。切り抜きの品質ではなく「どちらの世界か」が分かれば足りる。
"""

from __future__ import annotations

import threading
import time

import numpy as np
import torch

from .depth import pick_device, resize_unit


class PersonSegmenter:
    name = "lraspp"

    def __init__(self, size: tuple[int, int] = (512, 288), device: torch.device | None = None) -> None:
        from torchvision.models.segmentation import (
            LRASPP_MobileNet_V3_Large_Weights,
            lraspp_mobilenet_v3_large,
        )

        weights = LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
        self.person = weights.meta["categories"].index("person")
        self.device = device or pick_device()
        self.size = size
        self.model = lraspp_mobilenet_v3_large(weights=weights).eval().to(self.device)
        self.params = sum(p.numel() for p in self.model.parameters())
        mean, std = _stats(self.device)
        self._mean, self._std = mean, std

    @torch.inference_mode()
    def infer(self, rgb: np.ndarray) -> np.ndarray:
        x = (resize_unit(rgb, self.size, self.device) - self._mean) / self._std
        out = self.model(x)["out"]
        return out.softmax(1)[0, self.person].float().cpu().numpy()


def _stats(device: torch.device):
    from .depth import _stats as s
    return s(device)


class SegmentWorker:
    """深度ワーカーと同じ約束: 最新1枚だけ、間に合わなければ捨てる、描画は待たせない。"""

    def __init__(self, size: tuple[int, int] = (512, 288), ema: float = 0.55) -> None:
        self.ema = ema
        self.model: PersonSegmenter | None = None
        self._pending: np.ndarray | None = None
        self._result: np.ndarray | None = None
        self._seq = 0
        self._taken = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._smooth: np.ndarray | None = None
        self.last_infer_s = 0.0
        self.dropped = 0
        self.done = 0

    def start(self) -> "SegmentWorker":
        self.model = PersonSegmenter()
        print(f"segment[lraspp]: {self.model.params / 1e6:.2f} M params on {self.model.device}")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def submit(self, rgb: np.ndarray) -> None:
        with self._lock:
            if self._pending is not None:
                self.dropped += 1
            self._pending = rgb

    def latest(self) -> np.ndarray | None:
        with self._lock:
            if self._result is None or self._seq == self._taken:
                return None
            self._taken = self._seq
            return self._result

    def _loop(self) -> None:
        assert self.model is not None
        while not self._stop.is_set():
            with self._lock:
                frame, self._pending = self._pending, None
            if frame is None:
                time.sleep(0.002)
                continue
            t0 = time.perf_counter()
            m = self.model.infer(frame)
            self.last_infer_s = time.perf_counter() - t0
            # マットが毎フレーム揺れると、人物の輪郭で背景と現実が交互に入れ替わって
            # 縁がちらつく。深度と同じく時間方向に均す。
            if self._smooth is None or self._smooth.shape != m.shape:
                self._smooth = m
            else:
                self._smooth = self.ema * self._smooth + (1.0 - self.ema) * m
            with self._lock:
                self._result = self._smooth.astype(np.float32)
                self._seq += 1
                self.done += 1

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
