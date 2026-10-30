"""Parameter counts and GFLOPs of the paper's ablation variants, built with our implementation
-> results/variants.json

No training happens here. These are architecture facts we can check against the paper's tables
without any weights. Accuracy of the variants is *not* measured; the paper's values are shown as Reported.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdlib import efficiency as eff  # noqa: E402
from cdlib.models.tinycd import TinyCD  # noqa: E402
from cdlib.paths import RESULTS_DIR  # noqa: E402

# (id, description, constructor kwargs, paper table, paper params or None)
VARIANTS = [
    ("paper", "TinyCD as published (B4 slice, Mix, MAMB skips, PW-MLP classifier)", {}, "Tables 3/4/6", 285_128),
    ("sub_all", "Subtraction in skips and bottleneck", dict(fusion="sub"), "Table 6 / 7a", 282_939),
    ("sub_skips", "Subtraction in skips only (Mix in bottleneck)", dict(fusion="sub", fusion_bottleneck="mix"),
     "Table 7a", 284_004),
    ("sub_bottleneck", "Subtraction in bottleneck only (Mix in skips)", dict(fusion_bottleneck="sub"),
     "Table 7a", 284_063),
    ("cat_all", "Concatenation + conv in skips and bottleneck", dict(fusion="cat"), "Table 6 / 7b", 368_468),
    ("cat_skips", "Concatenation + conv in skips only", dict(fusion="cat", fusion_bottleneck="mix"),
     "Table 7b", 313_028),
    ("cat_bottleneck", "Concatenation + conv in bottleneck only", dict(fusion_bottleneck="cat"),
     "Table 7b", 340_568),
    ("no_classifier", "No final PW-MLP classifier (sliced w/o MLP)", dict(classifier=False), "Table 4", 282_438),
    ("no_skip", "No MAMB skip masks", dict(skip_masks=False), "Table 5", None),
    ("b0", "EfficientNet-B0 backbone", dict(backbone="efficientnet_b0"), "Table 9", 79_480),
    ("b1", "EfficientNet-B1 backbone", dict(backbone="efficientnet_b1"), "Table 9", 122_092),
    ("b2", "EfficientNet-B2 backbone", dict(backbone="efficientnet_b2"), "Table 9", 148_040),
    ("b3", "EfficientNet-B3 backbone", dict(backbone="efficientnet_b3"), "Table 9", 178_716),
]

if __name__ == "__main__":
    a = torch.rand(1, 3, 256, 256)
    rows = []
    for vid, desc, kw, table, paper_params in VARIANTS:
        net = TinyCD(**kw).eval()
        p = eff.params_total(net)
        g, _ = eff.gflops(net, a, a)
        rows.append(dict(id=vid, description=desc, config=net.config, channels=net.channels, params=p,
                         paper_table=table, paper_params=paper_params,
                         match=None if paper_params is None else p == paper_params,
                         diff=None if paper_params is None else p - paper_params, gflops=g))
        print(f"{vid:15s} params={p:>8,d} paper={paper_params}  match={rows[-1]['match']}  GFLOPs={g:.3f}")
    (RESULTS_DIR / "variants.json").write_text(json.dumps(rows, indent=1))
