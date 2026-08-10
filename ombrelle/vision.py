"""深度とマットを **1 本のスレッド**で回すワーカー。

なぜ 1 本にしたか (実測に基づく):

深度とセグメンテーションをそれぞれ専用スレッドに置いたところ、アプリが間欠的に
ハングするようになった。同じ条件で 4 回ずつ走らせて数えた結果:

    ワーカー1本 (深度のみ)      正常 4/4   ハング 0
    ワーカー1本 (セグのみ)      正常 4/4   ハング 0
    ワーカー2本 (深度+セグ)     正常 1/4   ハング 2/4

**torch-MPS の推論スレッドを 2 本、OpenGL のメインスレッドと並行させると壊れる。**
macOS では OpenGL も Metal の上で動くので、3 つの経路が同じ GPU を奪い合う。
(なお「描画が速すぎて推論が飢える」という最初の仮説は否定された。
 描画を 40fps に絞っても、無制限の 120fps でも、どちらでも起きた)

深度とマットは**同じフレームから作られ、同じ合成に食われる**ので、
1 本のスレッドで順に実行するのが素直でもある。代償は直列化 (19.6 + 8.3 ≒ 28ms)
だが、もともと描画より遅い前提の設計なので影響しない。

約束は他のワーカーと同じ:
  * submit() は最新フレームで上書きするだけ (キューに溜めない)
  * 間に合わなければ入力フレームは黙って捨てる
  * latest() は「前回より新しい結果」があるときだけ返す
  * 例外は握り潰さず記録して表示する
"""

from __future__ import annotations

import threading
import time

import numpy as np

from .depth import build
from .normalize import DepthNormalizer


class VisionWorker:
    def __init__(
        self,
        depth_kind: str = "teacher",
        ckpt: str = "checkpoints/student.pt",
        want_matte: bool = False,
        matte_ema: float = 0.55,
        depth_ema: float = 0.70,
    ) -> None:
        self.ckpt = ckpt
        self.kind = depth_kind
        self.want_matte = want_matte
        self.matte_ema = matte_ema
        self.depth_ema = depth_ema

        self._models: dict[str, object] = {}
        self._seg = None
        self._pending: np.ndarray | None = None
        self._depth: np.ndarray | None = None
        self._matte: np.ndarray | None = None
        self._matte_smooth: np.ndarray | None = None
        self._depth_smooth: np.ndarray | None = None
        self._seq = 0
        self._taken_depth = 0
        self._taken_matte = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self.norm = DepthNormalizer()
        self.depth_s = 0.0
        self.matte_s = 0.0
        self.dropped = 0
        self.done = 0
        self.error: str | None = None

    # ------------------------------------------------------------ モデル
    def _depth_model(self, kind: str):
        if kind not in self._models:
            self._models[kind] = build(kind, self.ckpt)
        return self._models[kind]

    def _segmenter(self):
        if self._seg is None:
            from .segment import PersonSegmenter
            self._seg = PersonSegmenter()
            print(f"segment[lraspp]: {self._seg.params / 1e6:.2f} M params on {self._seg.device}")
        return self._seg

    def start(self) -> "VisionWorker":
        # モデルの構築は同期で済ませる (最初のフレームまでに間に合わせる)
        if self.kind != "off":
            m = self._depth_model(self.kind)
            print(f"depth[{self.kind}]: {getattr(m, 'params', 0) / 1e6:.2f} M params on {m.device}")
        if self.want_matte:
            self._segmenter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    # ------------------------------------------------------------ 受け渡し
    def submit(self, rgb: np.ndarray) -> None:
        with self._lock:
            if self._pending is not None:
                self.dropped += 1
            self._pending = rgb

    def latest_depth(self) -> np.ndarray | None:
        with self._lock:
            if self._depth is None or self._seq == self._taken_depth:
                return None
            self._taken_depth = self._seq
            return self._depth

    def latest_matte(self) -> np.ndarray | None:
        with self._lock:
            if self._matte is None or self._seq == self._taken_matte:
                return None
            self._taken_matte = self._seq
            return self._matte

    # ------------------------------------------------------------ 本体
    def _loop(self) -> None:
        # スレッド内の例外は Python がそのスレッドだけを静かに終わらせる。
        # ワーカーが死んでも描画は前回の結果を使い続けるので気づけない。必ず記録する。
        try:
            self._run()
        except Exception as exc:                      # noqa: BLE001
            import traceback
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"[vision worker] 停止しました: {self.error}\n{traceback.format_exc()}", flush=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                frame, self._pending = self._pending, None
                kind, want = self.kind, self.want_matte
            if frame is None:
                time.sleep(0.002)
                continue

            depth = matte = None
            if kind != "off":
                t0 = time.perf_counter()
                depth = self.norm(self._depth_model(kind).infer(frame))
                # 深度マップそのものを時間方向に均す。
                #
                # DepthNormalizer が均しているのは min/max のレンジだけで、マップは素通り
                # だった。これが効くのは値ではなく**勾配の向き**である点に注意する。
                # 平坦な壁や机では深度の勾配がほぼ 0 になり、そこにノイズが乗ると
                # 「ほぼ 0 のベクトルの向き」= ほぼ乱数が毎フレーム変わる。
                # angle.frag はその向きを筆の向きに使うので、**平らな面の上で筆が
                # 回り続ける**ことになる。値のブレは小さくても向きのブレは最大になる。
                if self._depth_smooth is None or self._depth_smooth.shape != depth.shape:
                    self._depth_smooth = depth
                else:
                    self._depth_smooth = (self.depth_ema * self._depth_smooth
                                          + (1.0 - self.depth_ema) * depth)
                depth = self._depth_smooth.astype(np.float32)
                self.depth_s = time.perf_counter() - t0
            if want:
                t0 = time.perf_counter()
                m = self._segmenter().infer(frame)
                self.matte_s = time.perf_counter() - t0
                # マットが毎フレーム揺れると人物の輪郭で現実と絵が交互に入れ替わり、
                # 縁がちらつく。深度と同じ理由で時間方向に均す。
                if self._matte_smooth is None or self._matte_smooth.shape != m.shape:
                    self._matte_smooth = m
                else:
                    self._matte_smooth = (self.matte_ema * self._matte_smooth
                                          + (1.0 - self.matte_ema) * m)
                matte = self._matte_smooth.astype(np.float32)

            with self._lock:
                if depth is not None:
                    self._depth = depth
                if matte is not None:
                    self._matte = matte
                self._seq += 1
                self.done += 1

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
