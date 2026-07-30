"""自作の蒸留モデル。

設計の根拠:
  * エンコーダは MobileNetV3-Small (ImageNet 事前学習)。ゼロから学習させるだけの
    データを今日集められないので、事前学習済みの特徴に乗るのが正しい判断。
  * デコーダは 1x1 の横結合 + depthwise separable の融合だけ。パラメータの大半を
    エンコーダに置き、デコーダは軽く保つ。
  * 出力は入力の 1/2 解像度。深度は本来低周波な量で、レンダラ側は 1/4 拡大して
    線形補間で読む。フル解像度を出すのは帯域の無駄。
  * 学習目標は「教師の生出力」ではなく「画像ごとに頑健正規化した [0,1]」。
    レンダラが消費するのは正規化された相対深度なので、そこに直接合わせる。
    affine 不変損失を組むより実装が単純で、評価指標もそのまま解釈できる。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sep(cin: int, cout: int) -> nn.Sequential:
    """depthwise separable conv + BN + Hardswish"""
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False),
        nn.BatchNorm2d(cin),
        nn.Hardswish(inplace=True),
        nn.Conv2d(cin, cout, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.Hardswish(inplace=True),
    )


class StudentNet(nn.Module):
    # MobileNetV3-Small の features を stride が変わる位置で切る
    # 0:/2(16ch) 1:/4(16ch) 2-3:/8(24ch) 4-8:/16(48ch) 9-12:/32(576ch)
    SPLITS = [(0, 1), (1, 2), (2, 4), (4, 9), (9, 13)]
    ENC_CH = [16, 16, 24, 48, 576]

    def __init__(self, width: float = 1.0, pretrained: bool = True, dec: int = 48) -> None:
        super().__init__()
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        feats = mobilenet_v3_small(weights=weights).features
        self.stages = nn.ModuleList([nn.Sequential(*feats[a:b]) for a, b in self.SPLITS])

        c = max(8, int(dec * width))
        self.lat = nn.ModuleList([nn.Conv2d(ch, c, 1, bias=False) for ch in self.ENC_CH])
        self.fuse = nn.ModuleList([_sep(c, c) for _ in range(4)])
        self.head = nn.Sequential(_sep(c, c), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for st in self.stages:
            x = st(x)
            skips.append(x)
        y = self.lat[4](skips[4])
        for i in (3, 2, 1):
            s = self.lat[i](skips[i])
            y = F.interpolate(y, size=s.shape[-2:], mode="bilinear", align_corners=False)
            y = self.fuse[i](y + s)
        # 最後は 1/2 解像度まで上げる
        s = self.lat[0](skips[0])
        y = F.interpolate(y, size=s.shape[-2:], mode="bilinear", align_corners=False)
        y = self.fuse[0](y + s)
        return self.head(y)


# ---------------------------------------------------------------- 損失
def gradient_loss(pred: torch.Tensor, target: torch.Tensor, scales: int = 4) -> torch.Tensor:
    """多重スケールの勾配マッチング。

    L1 だけで学習すると輪郭が鈍る。深度の輪郭は筆触の向き(等深線に沿う)と
    オクルージョンの境界に直結するので、勾配を明示的に合わせる必要がある。
    """
    loss = pred.new_zeros(())
    p, t = pred, target
    for _ in range(scales):
        dpx = p[..., :, 1:] - p[..., :, :-1]
        dtx = t[..., :, 1:] - t[..., :, :-1]
        dpy = p[..., 1:, :] - p[..., :-1, :]
        dty = t[..., 1:, :] - t[..., :-1, :]
        loss = loss + (dpx - dtx).abs().mean() + (dpy - dty).abs().mean()
        if min(p.shape[-2:]) < 8:
            break
        p = F.avg_pool2d(p, 2)
        t = F.avg_pool2d(t, 2)
    return loss / scales


def distill_loss(pred: torch.Tensor, target: torch.Tensor, w_grad: float = 0.5) -> dict:
    l1 = (pred - target).abs().mean()
    lg = gradient_loss(pred, target)
    return {"loss": l1 + w_grad * lg, "l1": l1.detach(), "grad": lg.detach()}
