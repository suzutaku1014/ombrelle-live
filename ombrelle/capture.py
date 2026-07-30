"""カメラ取り込み。

設計の要点: 取り込みは専用スレッドで回し、**最新の1枚だけ**を保持する。
キューに溜めない理由は、絵の応答が遅れると「人が動いたら絵が傾ぐ」体験が壊れるため。
古いフレームを律儀に処理するより、捨てて最新に追いつく方が正しい。
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np


class Camera:
    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        mirror: bool = True,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.mirror = mirror

        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None  # RGB uint8
        self._stamp: float = 0.0
        self._seq: int = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.grabbed = 0

    def start(self) -> "Camera":
        # macOS では AVFoundation を明示した方が起動が速く、解像度指定も通りやすい
        cap = cv2.VideoCapture(self.index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise RuntimeError(
                f"カメラ {self.index} を開けません。"
                "macOS のカメラアクセス権限（システム設定 > プライバシーとセキュリティ > カメラ）"
                "でターミナルアプリを許可してください。"
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, 60)
        self._cap = cap
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.width
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        assert self._cap is not None
        while not self._stop.is_set():
            ok, bgr = self._cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            if self.mirror:
                # 自撮り像にする。フロー場も一緒に鏡像化されるので後段で辻褄が合う
                bgr = cv2.flip(bgr, 1)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._frame = rgb
                self._stamp = time.perf_counter()
                self._seq += 1
                self.grabbed += 1

    def latest(self) -> tuple[np.ndarray | None, float, int]:
        """(RGB frame, 取り込み時刻, 連番) を返す。連番で「新しいか」を判定できる。"""
        with self._lock:
            if self._frame is None:
                return None, 0.0, 0
            return self._frame, self._stamp, self._seq

    def wait_first(self, timeout: float = 10.0) -> np.ndarray:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            frame, _, _ = self.latest()
            if frame is not None:
                return frame
            time.sleep(0.02)
        raise RuntimeError("カメラから最初のフレームが来ません（権限か他アプリの占有を確認）")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
