"""Choose a fixed set of demo pairs by explicit rules and pre-render every model's output -> assets/demo/

The pairs are picked by rules over the full-test-set per-image results (no hand-picking), so the demo is not
cherry-picked. It includes a clear failure and a false alarm. Pre-rendered outputs let the app show every
pair even if live inference fails on presentation day.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdlib.data import load_pair, to_tensor  # noqa: E402
from cdlib.env import pick_device  # noqa: E402
from cdlib.metrics import confusion, per_image_f1, scores  # noqa: E402
from cdlib.models.registry import MODEL_KEYS, load_model  # noqa: E402
from cdlib.paths import DATA_DIR, DEMO_DIR, RESULTS_DIR  # noqa: E402


def pick():
    d = {k: np.load(RESULTS_DIR / f"per_image_{k}_test.npz") for k in MODEL_KEYS}
    names = d["tinycd"]["names"]
    c = {k: v["counts"] for k, v in d.items()}
    gt_frac = (c["tinycd"][:, 0] + c["tinycd"][:, 2]) / c["tinycd"].sum(1)
    f1 = {k: np.nan_to_num(per_image_f1(v), nan=-1) for k, v in c.items()}
    pred_px = {k: v[:, 0] + v[:, 1] for k, v in c.items()}
    order = np.argsort(names)  # deterministic tie-break by file name

    def best(score, mask):
        idx = order[mask[order]]
        return idx[np.argmax(score[idx])]

    picks = []
    used = set()

    def add(i, rule, title):
        if i in used:
            return
        used.add(i)
        picks.append((int(i), rule, title))

    add(best(f1["tinycd"], gt_frac > 0.15), "GT change > 15%, highest TinyCD F1", "Large new development")
    add(best(f1["tinycd"], (gt_frac > 0.01) & (gt_frac < 0.04)), "GT change 1-4%, highest TinyCD F1",
        "A few new buildings")
    spread = np.max([f1[k] for k in MODEL_KEYS], 0) - np.min([f1[k] for k in MODEL_KEYS], 0)
    add(best(spread, gt_frac > 0.01), "GT change > 1%, largest F1 spread between the three models",
        "Models disagree")
    add(best(-f1["tinycd"], gt_frac > 0.02), "GT change > 2%, lowest TinyCD F1", "TinyCD failure case")
    total_pred = sum(pred_px.values())
    add(best(-total_pred, gt_frac == 0), "No GT change, fewest changed pixels predicted by all models",
        "No change (clean)")
    add(best(pred_px["tinycd"].astype(float), gt_frac == 0), "No GT change, most pixels TinyCD predicts as change",
        "No change - TinyCD false alarm")
    return names, picks


if __name__ == "__main__":
    dev = pick_device("auto")
    names, picks = pick()
    if DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)
    DEMO_DIR.mkdir(parents=True)
    index = []
    for i, rule, title in picks:
        n = str(names[i])
        sub = DEMO_DIR / Path(n).stem
        sub.mkdir()
        for folder in ("A", "B", "label"):
            shutil.copy(DATA_DIR / folder / n, sub / f"{folder}.png")
        index.append({"name": n, "dir": sub.name, "title": title, "rule": rule, "models": {}})
    for key in MODEL_KEYS:
        model = load_model(key, dev)
        for entry in index:
            a, b, m = load_pair(entry["name"])
            prob = model.predict(to_tensor(a)[None].to(dev), to_tensor(b)[None].to(dev))[0, 0].cpu().numpy()
            Image.fromarray((prob * 255).round().astype(np.uint8)).save(DEMO_DIR / entry["dir"] / f"prob_{key}.png")
            entry["models"][key] = scores(confusion(prob > 0.5, m))
            if key == "tinycd":
                _, f = model.net.forward_with_intermediates(model.normalise(to_tensor(a)[None].to(dev)),
                                                            model.normalise(to_tensor(b)[None].to(dev)))
                np.savez_compressed(DEMO_DIR / entry["dir"] / "tinycd_masks.npz",
                                    **{k: f[k][0, 0].float().cpu().numpy().astype(np.float16)
                                       for k in ("M256", "M128", "M64")})
        del model
        torch.cuda.empty_cache()
    (DEMO_DIR / "index.json").write_text(json.dumps(index, indent=1))
    for e in index:
        print(f"{e['title']:32s} {e['name']:18s} " + "  ".join(f"{k}={v['f1']:.1f}" for k, v in e["models"].items()))
