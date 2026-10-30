# Vendored competitor code

The competitors use their official model code. ChangeFormer is included under MIT. BIT source is fetched
at setup time by `scripts/bootstrap_bit.py`, hash-checked and patched locally; it is excluded from publication.

| Folder | Source repository | Commit | Files | Licence |
|---|---|---|---|---|
| `bit/` | https://github.com/justchenhao/BIT_CD | `adcd7aea6f234586ffffdd4e9959404f96271711` | `models/networks.py`, `models/help_funcs.py`, `models/resnet.py` | No upstream licence file. Downloaded directly from the authors during setup for academic evaluation; source and weights are not republished here. |
| `changeformer/` | https://github.com/wgcban/ChangeFormer | `afd1b7ed640aa265a2c730de958416ae7356a2f9` | `models/ChangeFormer.py`, `models/ChangeFormerBaseNetworks.py`, `models/help_funcs.py`, `models/pixel_shuffel_up.py` | MIT (`changeformer/LICENSE`) |

## Patches (imports and initialisation only; no layer or forward-pass change)

Every patched line is marked `[vendor patch]` or is a relative-import rewrite.

| File | Change | Why |
|---|---|---|
| `bit/networks.py` | `import models` → `from . import resnet as models`; `from models.help_funcs` → `from .help_funcs` | package-relative imports |
| `bit/networks.py` | `resnet18/34/50(pretrained=True, …)` → `pretrained=False` | avoid downloading ImageNet weights at construction; the released checkpoint overwrites every one of these weights (`load_state_dict(strict=True)`) |
| `bit/resnet.py` | `torchvision.models.utils.load_state_dict_from_url` → `torch.hub.load_state_dict_from_url` | the old path was removed from torchvision; same function |
| `changeformer/ChangeFormer.py` | `from models.X import` → `from .X import` (3 lines) | package-relative imports |

You can check this yourself with `diff third_party/BIT_CD/models/networks.py cdlib/models/vendor/bit/networks.py`
(and the same for the other files).
