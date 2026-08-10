"""Oklab / Oklch 色空間。

なぜ要るか:

現在の色操作 (brush.frag の divide/compand/inject) は、luma
`dot(c, vec3(0.299, 0.587, 0.114))` を抜いた対立色平面で行っている。これは

  * ガンマ符号化された RGB 上の量であって、測光的な輝度ではない
  * 「色度ベクトルの長さ」は知覚的な彩度に比例しない (青は同じ長さでも鈍く見える)
  * 色相角も知覚的に等間隔ではない

という近似で、リアルタイムの意匠制御としては十分機能してきた。ただし
「人物と背景の彩度比」のような**領域間の比較**をするときは、この歪みが直接
判断を誤らせる。比較と測定には知覚軸を使う。

Oklab は L(明るさ) / a,b(対立色)、その極座標 Oklch は L / C(彩度) / h(色相角)。
完全な色外観モデル (CIECAM16 等) ではなく、順応や周辺の効果は扱わない。
それでも画像処理・色差・グラデーションの用途では HSL や生 RGB より
数値と直感が一致する。

係数は Björn Ottosson, "A perceptual color space for image processing" (2020)
https://bottosson.github.io/posts/oklab/ の原典そのまま。
**GLSL 版 (gl/shaders/brush.frag) と必ず同じ値を使うこと。**
片方だけ書き換えると A/B 比較が意味を失う。
"""

from __future__ import annotations

import numpy as np

# 線形 sRGB → LMS
_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
], dtype=np.float32)

# LMS の立方根 → Lab
_M2 = np.array([
    [0.2104542553,  0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050,  0.4505937099],
    [0.0259040371,  0.7827717662, -0.8086757660],
], dtype=np.float32)

# Lab → LMS の立方根
_M2_INV = np.array([
    [1.0,  0.3963377774,  0.2158037573],
    [1.0, -0.1055613458, -0.0638541728],
    [1.0, -0.0894841775, -1.2914855480],
], dtype=np.float32)

# LMS → 線形 sRGB
_M1_INV = np.array([
    [ 4.0767416621, -3.3077115913,  0.2309699292],
    [-1.2684380046,  2.6097574011, -0.3413193965],
    [-0.0041960863, -0.7034186147,  1.7076147010],
], dtype=np.float32)


# ---------------------------------------------------------------- 伝達関数
def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """sRGB → 線形 RGB。**絶対値に適用して符号を戻す**。

    sRGB の伝達関数は [0,1] の外では定義されていない。ところがシェーダ側の色は
    グレーディングで 1.0 を超えるし、色彩分割で成分が負にもなる。そこで
    「絶対値へ適用して符号を戻す」という拡張を使う (CSS Color 4 と同じ流儀)。

    正負で別の式を当てると、色域の外側で GLSL 版と数値が合わなくなる。
    実際 1 度それで食い違った (最大誤差 0.65)。両実装で同じ拡張を使うこと。
    """
    c = np.asarray(c, dtype=np.float32)
    a = np.abs(c)
    lo = a / 12.92
    hi = np.power((a + 0.055) / 1.055, 2.4)
    return (np.sign(c) * np.where(a <= 0.04045, lo, hi)).astype(np.float32)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float32)
    a = np.abs(c)
    lo = a * 12.92
    hi = 1.055 * np.power(a, 1.0 / 2.4) - 0.055
    return (np.sign(c) * np.where(a <= 0.0031308, lo, hi)).astype(np.float32)


# ---------------------------------------------------------------- Oklab
def linear_to_oklab(lin: np.ndarray) -> np.ndarray:
    lms = np.asarray(lin, dtype=np.float32) @ _M1.T
    # 色域外だと LMS が負になりうる。np.cbrt は負の実数根を返すので破綻しない
    return (np.cbrt(lms) @ _M2.T).astype(np.float32)


def oklab_to_linear(lab: np.ndarray) -> np.ndarray:
    lms_ = np.asarray(lab, dtype=np.float32) @ _M2_INV.T
    return ((lms_ ** 3) @ _M1_INV.T).astype(np.float32)


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    return linear_to_oklab(srgb_to_linear(rgb))


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    return linear_to_srgb(oklab_to_linear(lab))


def rgb8_to_oklab(rgb8: np.ndarray) -> np.ndarray:
    """uint8 の RGB 画像 (H, W, 3) → Oklab float32。"""
    return srgb_to_oklab(np.asarray(rgb8, dtype=np.float32) / 255.0)


# ---------------------------------------------------------------- Oklch
def oklab_to_oklch(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float32)
    a, b = lab[..., 1], lab[..., 2]
    return np.stack([lab[..., 0], np.hypot(a, b), np.arctan2(b, a)], axis=-1).astype(np.float32)


def oklch_to_oklab(lch: np.ndarray) -> np.ndarray:
    lch = np.asarray(lch, dtype=np.float32)
    C, h = lch[..., 1], lch[..., 2]
    return np.stack([lch[..., 0], C * np.cos(h), C * np.sin(h)], axis=-1).astype(np.float32)


def hue_distance(h1: float, h2: float) -> float:
    """円周上の最短距離 (0..π)。色相は角度なので単純な差を取ってはいけない。"""
    d = abs(float(h1) - float(h2)) % (2.0 * np.pi)
    return float(min(d, 2.0 * np.pi - d))


if __name__ == "__main__":
    # 往復誤差と既知の基準点。GLSL 版を書いたら同じ値で照合する。
    rng = np.random.default_rng(0)
    x = rng.random((2000, 3), dtype=np.float32)
    err = np.abs(oklab_to_srgb(srgb_to_oklab(x)) - x).max()
    print(f"sRGB 往復の最大誤差: {err:.2e}")

    for name, rgb in [("white", (1, 1, 1)), ("mid grey", (0.5, 0.5, 0.5)),
                      ("red", (1, 0, 0)), ("green", (0, 1, 0)), ("blue", (0, 0, 1))]:
        lab = srgb_to_oklab(np.array(rgb, dtype=np.float32))
        lch = oklab_to_oklch(lab)
        print(f"{name:9s} L={lch[0]:.4f} C={lch[1]:.4f} h={lch[2]:+.4f}")
