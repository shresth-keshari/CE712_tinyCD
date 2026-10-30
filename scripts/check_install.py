"""Verify shipped evidence and run one real CPU inference with every installed checkpoint."""
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import torch
from PIL import Image
from cdlib.data import to_tensor
from cdlib.models.registry import WEIGHTS, load_model
from cdlib.usgs import load_prepared, sha256
from verify_artifacts import verify


def main():
    torch.set_num_threads(4)
    verify(check_weights=True)
    demo = json.loads((ROOT / 'assets/demo/index.json').read_text())[0]
    folder = ROOT / 'assets/demo' / demo['dir']
    a, b = [np.array(Image.open(folder / n).convert('RGB')) for n in ('A.png', 'B.png')]
    for key in WEIGHTS:
        model = load_model(key, 'cpu', check_hash=True)
        prob = model.predict(to_tensor(a)[None], to_tensor(b)[None])[0, 0].numpy()
        assert prob.shape == (256, 256) and np.isfinite(prob).all()
        assert ((prob >= 0) & (prob <= 1)).all()
        print(f'{key}: verified checkpoint, frozen weights and real CPU inference', flush=True)
        del model
        gc.collect()
    _, _, valid, _ = load_prepared(ROOT / 'assets/powai/pair')
    saved = ROOT / 'assets/powai/predictions'
    report = json.loads((saved / 'report.json').read_text())
    assert report['pair_manifest_sha256'] == sha256(ROOT / 'assets/powai/pair/manifest.json')
    for key in WEIGHTS:
        assert report['models'][key]['probability_sha256'] == sha256(saved / f'prob_{key}.npz')
    print(f'Bundled Powai imagery and saved predictions verified ({valid.sum()} valid pixels).')


if __name__ == '__main__':
    main()
