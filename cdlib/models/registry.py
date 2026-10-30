"""One interface for every model: raw images in [0, 1] -> probability of change (B, 1, H, W).

| key           | model                                 | weights                                          | normalisation      | raw output                |
|---------------|---------------------------------------|--------------------------------------------------|--------------------|---------------------------|
| tinycd        | TinyCD, *our implementation*          | authors' LEVIR-CD checkpoint (`levir_best.pth`)  | ImageNet mean/std  | sigmoid probability       |
| bit           | BIT, official code (vendored)         | official LEVIR-CD checkpoint                     | mean = std = 0.5   | 2-class logits            |
| changeformer  | ChangeFormer V6, official (vendored)  | official LEVIR-CD checkpoint                     | mean = std = 0.5   | list of logits, last = full resolution |

For 2-class logits, P(change) = softmax(logits)[:, 1]; thresholding it at 0.5 is the same decision as the
`argmax` used by the BIT / ChangeFormer evaluators.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import torch
from torch import nn

from ..paths import WEIGHTS_DIR
from .tinycd import TinyCD, load_official_weights

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HALF = (0.5, 0.5, 0.5)

# SHA-256 of the checkpoints used for every number in results/ (verified by scripts/download.py).
WEIGHTS = {
    "tinycd": dict(
        path=WEIGHTS_DIR / "TinyCD_official" / "levir_best.pth",
        sha256="743645b87199c96bd0c3271ae5df6f0918b3f1d84618dfb9396843ca08ee4a3e",
        url="https://github.com/AndreaCodegoni/Tiny_model_4_CD/raw/main/pretrained_models/levir_best.pth",
        source="github.com/AndreaCodegoni/Tiny_model_4_CD (pretrained_models/levir_best.pth)",
    ),
    "bit": dict(
        path=WEIGHTS_DIR / "BIT_LEVIR" / "best_ckpt.pt",
        sha256="c159ba76143447f58c9f367ce8126a0014f2e4ba218cdb97cca173952c38cb3b",
        gdrive_id="1IVdF5a3e1_7DiSndtMkhpZuCSgDLLFcg",
        source="github.com/justchenhao/BIT_CD README (Google Drive), variant base_transformer_pos_s4_dd8_dedim8",
    ),
    "changeformer": dict(
        path=WEIGHTS_DIR / "ChangeFormer_LEVIR" / "best_ckpt.pt",
        sha256="db0dd783c3c3f27d02f55f24fa8199d2b2b47c422c7f7bd2e4d96e2fa65d69f2",
        url=("https://github.com/wgcban/ChangeFormer/releases/download/v0.1.0/"
             "CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_linear_ce_multi_train_True_"
             "multi_infer_False_shuffle_AB_False_embed_dim_256.zip"),
        source="github.com/wgcban/ChangeFormer release v0.1.0 (LEVIR-CD, embed_dim 256)",
    ),
}

INFO = {
    "tinycd": dict(label="TinyCD", paper_key="TinyCD",
                   desc="TinyCD - our implementation, authors' released LEVIR-CD weights"),
    "bit": dict(label="BIT", paper_key="BIT",
                desc="BIT (Chen, Qi & Shi, IEEE TGRS 2021) - official code and LEVIR-CD weights"),
    "changeformer": dict(label="ChangeFormer V6", paper_key="ChangeFormer",
                         desc="ChangeFormer V6 (Bandara & Patel, IGARSS 2022) - official code and LEVIR-CD weights"),
}
MODEL_KEYS = tuple(INFO)


def sha256(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


@dataclass
class ChangeModel:
    key: str
    net: nn.Module
    mean: tuple
    std: tuple
    output: str                       # "prob" | "logits2" | "list_logits2"
    meta: dict = field(default_factory=dict)

    def to(self, device):
        self.net.to(device)
        return self

    @property
    def device(self):
        return next(self.net.parameters()).device

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        m = torch.tensor(self.mean, device=x.device).view(1, 3, 1, 1)
        s = torch.tensor(self.std, device=x.device).view(1, 3, 1, 1)
        return (x - m) / s

    def raw_to_prob(self, out) -> torch.Tensor:
        if self.output == "list_logits2":
            out = out[-1]
        if self.output in ("logits2", "list_logits2"):
            return torch.softmax(out, dim=1)[:, 1:2]
        return out

    @torch.no_grad()
    def predict(self, a: torch.Tensor, b: torch.Tensor, **kw) -> torch.Tensor:
        """a, b: float (B, 3, H, W) in [0, 1] on the model's device -> P(change) (B, 1, H, W)."""
        self.net.eval()
        return self.raw_to_prob(self.net(self.normalise(a), self.normalise(b), **kw))


def load_model(key: str, device="cpu", check_hash: bool = False) -> ChangeModel:
    w = WEIGHTS[key]
    if not w["path"].exists():
        raise FileNotFoundError(f"{w['path']} missing - run `python scripts/download.py`")
    if check_hash and sha256(w["path"]) != w["sha256"]:
        raise RuntimeError(f"checksum mismatch for {w['path']}")

    if key == "tinycd":
        net = load_official_weights(TinyCD("efficientnet_b4"), w["path"])
        m = ChangeModel(key, net, IMAGENET_MEAN, IMAGENET_STD, "prob")
    elif key == "bit":
        from .vendor.bit.networks import BASE_Transformer
        # Configuration of BIT's `define_G('base_transformer_pos_s4_dd8_dedim8')`.
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                               with_pos="learned", enc_depth=1, dec_depth=8, decoder_dim_head=8)
        ck = torch.load(w["path"], map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model_G_state_dict"], strict=True)
        m = ChangeModel(key, net, HALF, HALF, "logits2")
    elif key == "changeformer":
        from .vendor.changeformer.ChangeFormer import ChangeFormerV6
        net = ChangeFormerV6(embed_dim=256)
        ck = torch.load(w["path"], map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model_G_state_dict"], strict=True)
        m = ChangeModel(key, net, HALF, HALF, "list_logits2")
    else:
        raise KeyError(key)
    m.net.eval()
    m.net.requires_grad_(False)
    m.meta = dict(INFO[key], weights_source=w["source"], sha256=w["sha256"])
    return m.to(device)
