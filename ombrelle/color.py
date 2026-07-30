"""照明の色かぶりの正規化 (グレーワールド)。

なぜ要るか:

絵の色は「影は寒色・光は暖色」という**相対的な**決め方をしている。ところがカメラの
ホワイトバランスは撮影環境で勝手に決まるので、電球色の部屋では入力が既に暖色に
転んでいる。そこへ暖色グレーディングを重ね、さらに彩度で増幅すると、
白い紙まで黄色くなる (実写で発生)。

絵の色の決定を照明から切り離すため、**先に色かぶりを外してから**グレーディングする。
グレーワールド仮説 (シーンの平均は無彩色) を部分的に (既定 75%) 適用する。
完全に適用しないのは、画面が単色で埋まっている場面 (緑の芝生一面など) で
過補正になるため。

これは意匠のためだけでなく、照明条件が変わっても同じ絵の法則が成立するという
ロバスト性そのものでもある。
"""

from __future__ import annotations

import cv2
import numpy as np

LW = np.array([0.299, 0.587, 0.114], dtype=np.float32)


class WhiteBalance:
    def __init__(self, strength: float = 0.75, ema: float = 0.90) -> None:
        self.strength = strength
        self.ema = ema
        self.gain = np.ones(3, dtype=np.float32)

    def update(self, rgb: np.ndarray) -> np.ndarray:
        """RGB uint8 → シェーダへ渡す除算係数 (輝度を変えない正規化済み)"""
        small = cv2.resize(rgb, (64, 36), interpolation=cv2.INTER_AREA)
        mean = small.reshape(-1, 3).mean(axis=0).astype(np.float32) / 255.0
        mean = np.maximum(mean, 1e-3)
        # 平均を無彩色へ寄せる係数。輝度は動かさないよう正規化する
        g = mean / float(mean @ LW)
        g = 1.0 + (g - 1.0) * self.strength
        g = g / float(g @ LW)
        self.gain = self.ema * self.gain + (1.0 - self.ema) * g
        return self.gain
