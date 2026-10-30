"""Run with:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdlib.data import EXPECTED_SIZES, read_list  # noqa: E402
from cdlib.metrics import bootstrap_f1, confusion, paired_bootstrap_delta_f1, scores  # noqa: E402
from cdlib.models.registry import WEIGHTS, load_model  # noqa: E402
from cdlib.models.tinycd import MixingBlock, TinyCD, count_params, interleave  # noqa: E402
from cdlib.paths import DATA_DIR  # noqa: E402

HAS_DATA = (DATA_DIR / "list" / "test.txt").exists()


# ----------------------------------------------------------------------------------------------- metrics
def test_metrics_hand_example():
    gt = np.array([[1, 1, 1, 0], [0, 0, 0, 0]], bool)
    pr = np.array([[1, 1, 0, 1], [0, 0, 0, 0]], bool)       # TP 2, FP 1, FN 1, TN 4
    c = confusion(pr, gt)
    assert c.tolist() == [[2, 1, 1, 4]]
    s = scores(c)
    assert s["precision"] == pytest.approx(100 * 2 / 3)
    assert s["recall"] == pytest.approx(100 * 2 / 3)
    assert s["f1"] == pytest.approx(100 * 2 / 3)
    assert s["iou"] == pytest.approx(50.0)
    assert s["oa"] == pytest.approx(75.0)


def test_metrics_sum_over_images_not_mean():
    # image 1: perfect on 1 changed pixel; image 2: 9 changed pixels all missed -> global recall 10%, not 50%.
    c = np.array([[1, 0, 0, 99], [0, 0, 9, 91]])
    assert scores(c)["recall"] == pytest.approx(10.0)


def test_all_negative_predictor():
    s = scores(np.array([0, 0, 5, 95]))
    assert s["f1"] == 0 and s["oa"] == pytest.approx(95.0)


def test_bootstrap_contains_estimate():
    rng = np.random.default_rng(1)
    c = np.stack([rng.integers(50, 100, 300), rng.integers(0, 20, 300), rng.integers(0, 20, 300),
                  rng.integers(1000, 2000, 300)], 1)
    lo, hi = bootstrap_f1(c, 500)
    assert lo < scores(c)["f1"] < hi
    d, dlo, dhi, _ = paired_bootstrap_delta_f1(c, c, 200)
    assert d == 0 and dlo == 0 and dhi == 0


# ----------------------------------------------------------------------------------------------- Mix block
def test_interleave_order():
    x = torch.arange(3.0).view(1, 3, 1, 1)
    y = 10 + torch.arange(3.0).view(1, 3, 1, 1)
    assert interleave(x, y).flatten().tolist() == [0, 10, 1, 11, 2, 12]


def test_mix_generalises_subtraction():
    torch.manual_seed(0)
    x, y = torch.randn(2, 7, 9, 9), torch.randn(2, 7, 9, 9)
    conv = MixingBlock(14, 7)._convmix[0]
    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.zero_()
        conv.weight[:, 0, 1, 1] = 1
        conv.weight[:, 1, 1, 1] = -1
        assert torch.allclose(conv(interleave(x, y)), x - y, atol=1e-6)


# ----------------------------------------------------------------------------------------------- architecture
@pytest.mark.parametrize("kw,expected", [
    ({}, 285_128),                                              # Tables 3/4/6
    (dict(fusion="sub"), 282_939),                              # Table 6
    (dict(fusion="cat"), 368_468),                              # Table 6
    (dict(fusion="sub", fusion_bottleneck="mix"), 284_004),     # Table 7a
    (dict(fusion_bottleneck="sub"), 284_063),                   # Table 7a
    (dict(fusion="cat", fusion_bottleneck="mix"), 313_028),     # Table 7b
    (dict(fusion_bottleneck="cat"), 340_568),                   # Table 7b
    (dict(classifier=False), 282_438),                          # Table 4
    (dict(backbone="efficientnet_b3"), 178_716),                # Table 9
])
def test_param_counts_match_paper(kw, expected):
    assert count_params(TinyCD(**kw)) == expected


def test_forward_shapes_all_variants():
    a = torch.rand(2, 3, 256, 256)
    for kw in ({}, dict(skip_masks=False), dict(classifier=False), dict(fusion="sub")):
        out = TinyCD(**kw).eval()(a, a)
        assert out.shape == (2, 1, 256, 256) and 0 <= out.min() and out.max() <= 1


# ----------------------------------------------------------------------------------------------- weights
@pytest.mark.parametrize("key", list(WEIGHTS))
def test_official_weights_load_strictly(key):
    if not WEIGHTS[key]["path"].exists():
        pytest.skip("weights not downloaded")
    m = load_model(key, "cpu", check_hash=True)          # load_state_dict(strict=True) inside
    assert not m.net.training
    assert not any(p.requires_grad for p in m.net.parameters())
    p = m.predict(torch.rand(1, 3, 256, 256), torch.rand(1, 3, 256, 256))
    assert p.shape == (1, 1, 256, 256)
    if key == "tinycd":
        assert count_params(m.net) == 285_128


def test_tiled_inference_matches_direct():
    if not WEIGHTS["tinycd"]["path"].exists():
        pytest.skip("weights not downloaded")
    from cdlib.data import to_tensor
    from cdlib.inference import predict_pair
    m = load_model("tinycd", "cpu")
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 255, (256, 256, 3), np.uint8), rng.integers(0, 255, (256, 256, 3), np.uint8)
    tiled, _ = predict_pair(m, a, b)
    direct = m.predict(to_tensor(a)[None], to_tensor(b)[None])[0, 0].numpy()
    assert np.allclose(tiled, direct, atol=1e-5)
    big, _ = predict_pair(m, np.zeros((300, 520, 3), np.uint8), np.zeros((300, 520, 3), np.uint8))
    assert big.shape == (300, 520)


# ----------------------------------------------------------------------------------------------- data
@pytest.mark.skipif(not HAS_DATA, reason="dataset not downloaded")
def test_split_sizes():
    for s, n in EXPECTED_SIZES.items():
        assert len(read_list(s)) == n
