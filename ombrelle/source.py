"""入力ソースの抽象化。

`Camera` と同じインターフェース (latest / wait_first / stop) を持つ差し替え可能な入力。
  cam:0          … Web カメラ
  path/to.mp4    … 動画ファイル(ループ再生)
  path/to.jpg    … 静止画(深度やグレーディングの検証に最適)
  synthetic      … 手続き生成の動くシーン

合成シーンを用意した理由は、カメラ権限が無い環境でもパイプライン全体
(フロー → 深度 → 筆触 → オクルージョン)を検証できるようにするため。
奥行きの手がかり(サイズ・重なり・地面との接地)と、周期的な動き(歩行と腕の上げ下げ)を
意図的に入れてある。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .capture import Camera


class _ThreadedSource:
    """latest() が「最新の1枚」を返す共通の器。"""

    def __init__(self, width: int, height: int) -> None:
        self.width, self.height = width, height
        self._frame: np.ndarray | None = None
        self._stamp = 0.0
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _publish(self, rgb: np.ndarray) -> None:
        with self._lock:
            self._frame = rgb
            self._stamp = time.perf_counter()
            self._seq += 1

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:  # pragma: no cover - サブクラスが実装
        raise NotImplementedError

    def latest(self) -> tuple[np.ndarray | None, float, int]:
        with self._lock:
            if self._frame is None:
                return None, 0.0, 0
            return self._frame, self._stamp, self._seq

    def wait_first(self, timeout: float = 10.0) -> np.ndarray:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            f, _, _ = self.latest()
            if f is not None:
                return f
            time.sleep(0.01)
        raise RuntimeError("入力ソースから最初のフレームが来ません")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


class VideoSource(_ThreadedSource):
    def __init__(self, path: str, width: int = 1280, height: int = 720, fps: float = 30.0) -> None:
        self.path = path
        self.fps = fps
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"動画/画像を開けません: {path}")
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if src_fps and src_fps > 1:
            self.fps = float(src_fps)
        self._cap = cap
        super().__init__(width, height)

    def _loop(self) -> None:
        period = 1.0 / max(self.fps, 1.0)
        while not self._stop.is_set():
            ok, bgr = self._cap.read()
            if not ok:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, bgr = self._cap.read()
                if not ok:
                    break
            bgr = cv2.resize(bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)
            self._publish(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            time.sleep(period)
        self._cap.release()


class ImageSource(_ThreadedSource):
    """静止画。動きが無いのでフローは 0、深度とグレーディングの検証用。"""

    def __init__(self, path: str, width: int = 1280, height: int = 720) -> None:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"画像を開けません: {path}")
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
        self._rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        super().__init__(width, height)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._publish(self._rgb)
            time.sleep(1.0 / 30.0)


class SyntheticSource(_ThreadedSource):
    """奥行きの手がかりと周期的な動きを持つ手続き生成シーン。"""

    def __init__(self, width: int = 1280, height: int = 720, fps: float = 60.0) -> None:
        self.fps = fps
        super().__init__(width, height)
        self._bg = self._background(width, height)

    @staticmethod
    def _background(w: int, h: int) -> np.ndarray:
        img = np.zeros((h, w, 3), np.float32)
        yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
        horizon = 0.52
        sky = np.clip((horizon - yy) / horizon, 0.0, 1.0)
        ground = np.clip((yy - horizon) / (1.0 - horizon), 0.0, 1.0)
        # 空: 地平は白く、天頂は青い
        img += sky * np.array([0.42, 0.58, 0.86], np.float32)
        img += (1.0 - np.clip(sky * 3.0, 0, 1)) * (yy < horizon) * np.array([0.95, 0.93, 0.90], np.float32)
        # 地面: 遠いほど明るく霞む
        img += ground * np.array([0.30, 0.38, 0.22], np.float32)
        img += (1.0 - ground) * (yy >= horizon) * np.array([0.62, 0.66, 0.52], np.float32)
        # 粒状のテクスチャ(オプティカルフローが掴む手がかり)
        rng = np.random.default_rng(7)
        grain = cv2.GaussianBlur(rng.random((h, w), dtype=np.float32), (0, 0), 2.0)
        img *= 0.85 + 0.30 * grain[..., None]
        # 奥行きの手がかり: 遠いほど小さく、地平線に近く接地する柱
        for i, (fx, scale) in enumerate([(0.10, 0.30), (0.26, 0.18), (0.45, 0.11),
                                          (0.68, 0.20), (0.88, 0.34)]):
            ph = int(scale * h)
            pw = max(4, int(0.035 * scale / 0.30 * w))
            cx = int(fx * w)
            base = int((horizon + 0.02 + 0.42 * scale) * h)
            shade = 0.28 + 0.10 * i
            cv2.rectangle(img, (cx - pw // 2, base - ph), (cx + pw // 2, base),
                          (shade * 0.9, shade, shade * 0.7), -1)
            cv2.ellipse(img, (cx, base - ph), (int(pw * 1.9), int(ph * 0.22)),
                        0, 0, 360, (0.18, 0.34, 0.16), -1)
        return np.clip(img, 0, 1)

    def _figure(self, img: np.ndarray, t: float) -> None:
        h, w = img.shape[:2]
        # 左右に歩き、周期的に腕を上げる
        walk = 0.5 + 0.28 * np.sin(2 * np.pi * t / 11.0)
        cx = int(walk * w)
        base = int(0.94 * h)
        bh = int(0.46 * h)
        bw = int(0.055 * w)
        skin = (0.72, 0.62, 0.55)
        cloth = (0.22, 0.24, 0.34)
        # 胴
        cv2.ellipse(img, (cx, base - bh // 2), (bw, bh // 2), 0, 0, 360, cloth, -1)
        # 頭
        cv2.circle(img, (cx, base - bh - int(0.045 * h)), int(0.042 * h), skin, -1)
        # 腕: 4秒周期で上げ下げ
        raise_amt = 0.5 + 0.5 * np.sin(2 * np.pi * t / 4.0)
        for side in (-1, 1):
            ang = np.deg2rad(20 + 130 * raise_amt) * side
            L = int(0.20 * h)
            sx = cx + side * bw
            sy = base - bh + int(0.03 * h)
            ex = int(sx + L * np.sin(ang))
            ey = int(sy - L * np.cos(ang) * raise_amt - L * 0.15)
            cv2.line(img, (sx, sy), (ex, ey), skin, max(3, int(0.014 * h)))
            cv2.circle(img, (ex, ey), int(0.016 * h), skin, -1)

    def _loop(self) -> None:
        period = 1.0 / self.fps
        t0 = time.perf_counter()
        while not self._stop.is_set():
            t = time.perf_counter() - t0
            img = self._bg.copy()
            self._figure(img, t)
            self._publish((np.clip(img, 0, 1) * 255).astype(np.uint8))
            time.sleep(period)


def open_source(spec: str, width: int, height: int, mirror: bool = True):
    """spec から入力ソースを作る。"""
    if spec == "synthetic":
        return SyntheticSource(width, height).start()
    if spec.startswith("cam:"):
        return Camera(int(spec[4:]), width, height, mirror=mirror).start()
    p = Path(spec)
    if not p.exists():
        raise RuntimeError(f"入力が見つかりません: {spec}")
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
        return ImageSource(str(p), width, height).start()
    return VideoSource(str(p), width, height).start()
