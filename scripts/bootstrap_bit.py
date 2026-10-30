"""Fetch pinned upstream BIT code at installation time; do not republish unlicensed source."""
import hashlib
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'cdlib/models/vendor/bit'
COMMIT = 'adcd7aea6f234586ffffdd4e9959404f96271711'
FILES = {
    'networks.py': 'a501801eee4d82410db6d188cb814cb92c5fcee7cfc3c9a95c87202c822881d5',
    'help_funcs.py': '7f9cb1f860a7568099b4f92d222f47f9f0206d12a08704839cda86fd7b975dd8',
    'resnet.py': 'c306949839ae411bfe472904fd06ad7bab117f19953f84c352466df233e62de2',
}


def patch(name, text):
    if name == 'networks.py':
        text = text.replace('import models\n', 'from . import resnet as models\n')
        text = text.replace('from models.help_funcs import', 'from .help_funcs import')
        text = text.replace('pretrained=True,', 'pretrained=False,  # [vendor patch] ImageNet init not needed: checkpoint overwrites it')
    elif name == 'resnet.py':
        text = text.replace('from torchvision.models.utils import load_state_dict_from_url',
                            'from torch.hub import load_state_dict_from_url  # [vendor patch] moved in torchvision >= 0.13')
    compile(text, name, 'exec')
    return text


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for name, digest in FILES.items():
        url = f'https://raw.githubusercontent.com/justchenhao/BIT_CD/{COMMIT}/models/{name}'
        response = subprocess.run(
            ['curl', '--fail', '--location', '--silent', '--show-error', '--retry', '3',
             '--retry-all-errors', '--connect-timeout', '30', '--max-time', '120', url],
            check=True, capture_output=True,
        )
        data = response.stdout
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f'Upstream BIT checksum mismatch: {name}')
        desired = patch(name, data.decode('utf-8'))
        path = DEST / name
        if path.exists():
            if path.read_text(encoding='utf-8') != desired:
                raise ValueError(f'{path} contains local changes; refusing to overwrite them')
        else:
            path.write_text(desired, encoding='utf-8', newline='\n')
        print(f'BIT source verified: {name}')


if __name__ == '__main__':
    main()
