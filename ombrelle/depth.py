"""単眼深度推定。

teacher (既製の Depth Anything V2 Small) と student (自作の蒸留モデル) を
**同一インターフェース**で差し替えられるようにしてある。実行中に `d` キーで
切り替えて、絵の見え方と FPS を同じ画面で比べられる。これが蒸留の効果を語る土台。

前処理は transformers の ImageProcessor を使わず cv2 で自前実装している。
理由は、ImageProcessor が PIL を経由して 1 フレームあたり数 ms を食うため。
リアルタイム経路では前処理も測定対象。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .normalize import DepthNormalizer

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TEACHER_ID = "depth-anything/Depth-Anything-V2-Small-hf"
# DINOv2 は patch 14 なので入力の縦横は 14 の倍数でなければならない。
# 518x518 が既定だが、16:9 に近い 392x224 (28x16 patch) で十分かつ大幅に軽い。
TEACHER_SIZE = (392, 224)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def preprocess(rgb: np.ndarray, size: tuple[int, int], device: torch.device) -> torch.Tensor:
    """RGB uint8 (H,W,3) → 正規化済み (1,3,h,w) テンソル"""
    w, h = size
    img = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    x = img.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))[None]
    return torch.from_numpy(np.ascontiguousarray(x)).to(device)


class TeacherDepth:
    """既製の Depth Anything V2 Small。蒸留の教師でもあり、比較の基準でもある。"""

    name = "teacher"

    def __init__(self, size: tuple[int, int] = TEACHER_SIZE, device: torch.device | None = None) -> None:
        from transformers import AutoModelForDepthEstimation

        self.device = device or pick_device()
        self.size = size
        self.model = AutoModelForDepthEstimation.from_pretrained(TEACHER_ID)
        self.model.eval().to(self.device)
        self.params = sum(p.numel() for p in self.model.parameters())

    @torch.inference_mode()
    def infer(self, rgb: np.ndarray) -> np.ndarray:
        x = preprocess(rgb, self.size, self.device)
        out = self.model(pixel_values=x).predicted_depth  # (1,h,w) 相対逆深度
        if out.dim() == 4:
            out = out[:, 0]
        return out[0].float().cpu().numpy()


class StudentDepth:
    """自作の蒸留モデル。train/distill.py が吐いたチェックポイントを読む。"""

    name = "student"

    def __init__(self, ckpt: str, device: torch.device | None = None) -> None:
        from train.student import StudentNet

        path = Path(ckpt)
        if not path.exists():
            raise FileNotFoundError(
                f"student のチェックポイントがありません: {ckpt}\n"
                "先に  uv run python -m train.collect  →  uv run python -m train.distill  を実行してください"
            )
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.device = device or pick_device()
        self.size = tuple(blob.get("size", (384, 224)))
        self.model = StudentNet(width=blob.get("width", 1.0))
        self.model.load_state_dict(blob["state_dict"])
        self.model.eval().to(self.device)
        self.params = sum(p.numel() for p in self.model.parameters())

    @torch.inference_mode()
    def infer(self, rgb: np.ndarray) -> np.ndarray:
        x = preprocess(rgb, self.size, self.device)
        out = self.model(x)
        if out.dim() == 4:
            out = out[:, 0]
        return out[0].float().cpu().numpy()


def build(kind: str, ckpt: str = "checkpoints/student.pt"):
    if kind == "teacher":
        return TeacherDepth()
    if kind == "student":
        return StudentDepth(ckpt)
    raise ValueError(kind)


class DepthWorker:
    """推論を専用スレッドで回す。

    描画を絶対に待たせない、という一点のために存在する。
      * submit() は最新フレームで上書きするだけ(キューに溜めない)
      * 推論が間に合わなければ入力フレームは黙って捨てる
      * latest() は「前回より新しい結果」があるときだけ返す
    """

    def __init__(self, kind: str = "teacher", ckpt: str = "checkpoints/student.pt") -> None:
        self.ckpt = ckpt
        self.kind = kind
        self._models: dict[str, object] = {}
        self._pending: np.ndarray | None = None
        self._result: np.ndarray | None = None
        self._result_seq = 0
        self._taken_seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.norm = DepthNormalizer()
        self.last_infer_s = 0.0
        self.dropped = 0
        self.done = 0
        self.error: str | None = None

    def _model(self, kind: str):
        if kind not in self._models:
            self._models[kind] = build(kind, self.ckpt)
        return self._models[kind]

    def start(self) -> "DepthWorker":
        # 起動時のモデル構築は同期で行う(最初のフレームまでに間に合わせる)
        m = self._model(self.kind)
        print(f"depth[{self.kind}]: {getattr(m, 'params', 0) / 1e6:.2f} M params on {m.device}")
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
            if self._result is None or self._result_seq == self._taken_seq:
                return None
            self._taken_seq = self._result_seq
            return self._result

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                frame, self._pending = self._pending, None
                kind = self.kind
            if frame is None:
                time.sleep(0.002)
                continue
            try:
                model = self._model(kind)
            except Exception as exc:  # student が無い等
                self.error = str(exc)
                self.kind = "teacher"
                continue
            t0 = time.perf_counter()
            raw = model.infer(frame)
            self.last_infer_s = time.perf_counter() - t0
            depth = self.norm(raw)
            with self._lock:
                self._result = depth
                self._result_seq += 1
                self.done += 1

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
