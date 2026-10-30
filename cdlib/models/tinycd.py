"""TinyCD (Codegoni et al., 2023, Section 3), written for this project.

Siamese U-Net::

    I1, I2 -> MAMB@256 (on the raw RGB images) ------------------------------------- M256 --+
    I1, I2 -> EfficientNet-B4 blocks 0-1 (shared weights) -> MAMB@128 -------------- M128 --+  skip masks
                          `-> block 2 -> MAMB@64 ---------------------------------- M64  --+
                                `-> block 3 -> Mix (bottleneck) -> U (56 ch, 32x32)          |
    U -> Up(x2) * M64 -> Up(x2) * M128 -> Up(x2) * M256 -> PW-MLP classifier + sigmoid -> P(change)

* **Mix** (Sec. 3.3.1): interleave the channels of the two images ``[X0, Y0, X1, Y1, ...]`` and apply a grouped
  3x3 convolution with ``groups = C``. Each kernel sees exactly one channel at t1 and the same channel at t2.
* **PW-MLP** (Sec. 3.3.3): a stack of 1x1 convolutions + PReLU, i.e. the same small MLP applied to every pixel.
* **MAMB** (Sec. 3.3.2) = Mix -> PW-MLP down to one channel: a spatial attention mask.
* **Up block** (Sec. 3.4): bilinear x2 upsampling, multiply by the skip mask, depth-wise separable conv.

Module/attribute names deliberately match the authors' code so that their released checkpoint loads with
``strict=True``: every key and every tensor shape must agree, which verifies the architecture.
Where the paper text and the code differ (layer order inside Mix/Up blocks), we follow the code, because it
produced the published weights.

Variant switches exist only to (a) count parameters of the paper's ablation variants and (b) run clearly
labelled inference-time probes. No ablation weights were released, so ablation *accuracy* is never measured here.
"""
from __future__ import annotations

from typing import Optional

import torch
import torchvision
from torch import Tensor, nn


class PixelwiseLinear(nn.Module):
    """PW-MLP: 1x1 conv + PReLU per layer; the last activation can be replaced (e.g. Sigmoid)."""

    def __init__(self, fin: list[int], fout: list[int], last_activation: Optional[nn.Module] = None):
        super().__init__()
        n = len(fin)
        self._linears = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(fin[i], fout[i], kernel_size=1, bias=True),
                nn.PReLU() if i < n - 1 or last_activation is None else last_activation,
            )
            for i in range(n)
        ])

    def forward(self, x: Tensor) -> Tensor:
        return self._linears(x)


def interleave(x: Tensor, y: Tensor) -> Tensor:
    """[X0, X1, ...], [Y0, Y1, ...] -> [X0, Y0, X1, Y1, ...] along the channel axis (Eq. 1)."""
    return torch.stack((x, y), dim=2).reshape(x.shape[0], -1, x.shape[2], x.shape[3])


class MixingBlock(nn.Module):
    """Mix (Eq. 2): grouped 3x3 conv over interleaved channels, then PReLU and InstanceNorm.

    Parameters: C kernels of size 2x3x3 plus C biases = C * (2*3*3) + C, plus one PReLU slope.
    """

    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self._convmix = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, groups=ch_out, padding=1),
            nn.PReLU(),
            nn.InstanceNorm2d(ch_out),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self._convmix(interleave(x, y))


class Subtraction(nn.Module):
    """Ablation (paper Tables 6-7): plain X - Y instead of Mix. No parameters."""

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return x - y


class ConcatConv(nn.Module):
    """Ablation (paper Tables 6-7): concatenate [X, Y] and apply a full 3x3 conv 2C -> C (+ PReLU, InstanceNorm).

    Parameters: C kernels of size 2C x 3 x 3 plus C biases = C * (2C*3*3) + C, plus one PReLU slope.
    """

    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self._conv = nn.Sequential(nn.Conv2d(ch_in, ch_out, 3, padding=1), nn.PReLU(), nn.InstanceNorm2d(ch_out))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self._conv(torch.cat((x, y), dim=1))


def fusion_block(kind: str, ch_in: int, ch_out: int) -> nn.Module:
    return {"mix": MixingBlock, "cat": ConcatConv}[kind](ch_in, ch_out) if kind != "sub" else Subtraction()


class MixingMaskAttentionBlock(nn.Module):
    """MAMB: fusion (Mix, or an ablation alternative) followed by a PW-MLP down to one channel -> spatial mask."""

    def __init__(self, ch_in: int, ch_out: int, fin: list[int], fout: list[int], fusion: str = "mix"):
        super().__init__()
        self._mixing = fusion_block(fusion, ch_in, ch_out)
        self._linear = PixelwiseLinear(fin, fout)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self._linear(self._mixing(x, y))


class UpMask(nn.Module):
    """Up block: bilinear upsampling, optional multiplication by a skip mask, depth-wise separable conv."""

    def __init__(self, scale_factor: float, nin: int, nout: int):
        super().__init__()
        self._upsample = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)
        self._convolution = nn.Sequential(
            nn.Conv2d(nin, nin, 3, 1, groups=nin, padding=1),   # depth-wise 3x3
            nn.PReLU(),
            nn.InstanceNorm2d(nin),
            nn.Conv2d(nin, nout, kernel_size=1, stride=1),       # point-wise 1x1
            nn.PReLU(),
            nn.InstanceNorm2d(nout),
        )

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        x = self._upsample(x)
        if y is not None:
            x = x * y
        return self._convolution(x)


def efficientnet_slice(name: str = "efficientnet_b4", last_block: int = 3) -> nn.ModuleList:
    """First `last_block + 1` stages of a torchvision EfficientNet (`features[0..3]` for TinyCD)."""
    feats = getattr(torchvision.models, name)(weights=None).features
    return nn.ModuleList([feats[i] for i in range(last_block + 1)])


class TinyCD(nn.Module):
    """TinyCD with switches for the paper's ablation variants.

    fusion:            "mix" (paper), "sub" or "cat" (Table 6) - used in the skips and the bottleneck
    fusion_bottleneck: overrides `fusion` in the bottleneck only (Table 7)
    skip_masks:        True (paper) or False (Table 5, decoder without MAMB masks)
    classifier:        True (paper) or False (Table 4, last Up block outputs one channel -> sigmoid)
    backbone:          any torchvision EfficientNet; b4 is the paper's choice (Table 9 compares others)
    """

    def __init__(self, backbone: str = "efficientnet_b4",
                 fusion: str = "mix", skip_masks: bool = True, classifier: bool = True,
                 fusion_bottleneck: Optional[str] = None):
        super().__init__()
        fusion_bottleneck = fusion_bottleneck or fusion
        self.config = dict(backbone=backbone, fusion=fusion, fusion_bottleneck=fusion_bottleneck,
                           skip_masks=skip_masks, classifier=classifier)
        self._backbone = efficientnet_slice(backbone)

        # Probe channel widths of backbone stages 1, 2, 3 (24/32/56 for B4).
        # eval() so the probe does not touch BatchNorm running statistics of a pre-trained backbone.
        self._backbone.eval()
        with torch.no_grad():
            x, chans = torch.zeros(1, 3, 64, 64), []
            for i, layer in enumerate(self._backbone):
                x = layer(x)
                if i > 0:
                    chans.append(x.shape[1])
        c1, c2, c3 = chans
        self.channels = dict(c128=c1, c64=c2, c32=c3)

        def mamb(c):
            return MixingMaskAttentionBlock(2 * c, c, [c, c // 2, c // 4], [c // 2, c // 4, 1], fusion)

        self.skip_masks = skip_masks
        bottleneck = fusion_block(fusion_bottleneck, 2 * c3, c3)
        if skip_masks:
            self._first_mix = MixingMaskAttentionBlock(6, 3, [3, 10, 5], [10, 5, 1], fusion)
            self._mixing_mask = nn.ModuleList([mamb(c1), mamb(c2), bottleneck])
        else:
            self._mixing_mask = nn.ModuleList([bottleneck])

        last_out = 32 if classifier else 1
        self._up = nn.ModuleList([UpMask(2, c3, 64), UpMask(2, 64, 64), UpMask(2, 64, last_out)])
        self._classify = (PixelwiseLinear([32, 16, 8], [16, 8, 1], nn.Sigmoid()) if classifier else nn.Sigmoid())

    # ----------------------------------------------------------------------------------------------------------
    def encode(self, ref: Tensor, test: Tensor) -> dict[str, Tensor]:
        """Returns the skip masks M256, M128, M64 (if enabled) and the bottleneck feature U32."""
        out = {}
        if self.skip_masks:
            out["M256"] = self._first_mix(ref, test)
        for num, layer in enumerate(self._backbone):
            ref, test = layer(ref), layer(test)
            if num == 0:
                continue
            if not self.skip_masks:
                if num == 3:
                    out["U32"] = self._mixing_mask[0](ref, test)
                continue
            key = {1: "M128", 2: "M64", 3: "U32"}[num]
            out[key] = self._mixing_mask[num - 1](ref, test)
        return out

    def decode(self, f: dict[str, Tensor], mask_override: Optional[dict] = None) -> Tensor:
        """Decoder. `mask_override` (probes only) maps a mask name to "ones" (mask removed) or a tensor."""
        mask_override = mask_override or {}
        x = f["U32"]
        for up, key in zip(self._up, ("M64", "M128", "M256")):
            m = f.get(key)
            if key in mask_override:
                ov = mask_override[key]
                m = None if (isinstance(ov, str) and ov == "ones") else ov
            x = up(x, m)
        return x

    def forward(self, ref: Tensor, test: Tensor, mask_override: Optional[dict] = None) -> Tensor:
        return self._classify(self.decode(self.encode(ref, test), mask_override))

    @torch.no_grad()
    def forward_with_intermediates(self, ref: Tensor, test: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        f = self.encode(ref, test)
        return self._classify(self.decode(f)), f


def load_official_weights(model: TinyCD, path) -> TinyCD:
    """Load the authors' checkpoint with strict key/shape checking. Raises if anything differs."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)
    return model


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
