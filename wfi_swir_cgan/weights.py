"""Locate, download and load the pretrained generator weights.

Weights are looked up in the following order:

1. An explicit path passed by the user (``weights_path=...``).
2. The directory given by the ``WFI_SWIR_CGAN_WEIGHTS_DIR`` environment variable.
3. The ``weights/`` directory inside the installed package (useful for a
   cloned repository or an editable install).
4. The user cache directory (``~/.cache/wfi_swir_cgan`` by default, or
   ``$XDG_CACHE_HOME/wfi_swir_cgan``). If the file is missing it is
   downloaded from the GitHub release defined by ``WEIGHTS_BASE_URL``.

Files that are Git LFS pointers (a clone made without ``git-lfs`` installed)
are ignored.
"""

import os
from pathlib import Path

import numpy as np
import torch

try:  # NumPy >= 2
    from numpy._core.multiarray import scalar as _np_scalar
except ImportError:  # NumPy 1.x
    from numpy.core.multiarray import scalar as _np_scalar

# Target bands available, mapped to the checkpoint file names.
WEIGHT_FILES = {
    'B11': 'best_model_B11.pth',
    'B12': 'best_model_B12.pth',
}

# The ``.pth`` files (~200 MB each) are not shipped in the wheel (PyPI limits files
# to 100 MB). They are downloaded from the assets of this GitHub release, which must
# keep the names listed in ``WEIGHT_FILES``. New weights go into a new release
# (``weights-v2``...), so older package versions keep downloading their own weights.
WEIGHTS_BASE_URL = os.environ.get(
    'WFI_SWIR_CGAN_WEIGHTS_URL',
    'https://github.com/marujore/wfi-swir-cgan/releases/download/weights-v1',
)

_LFS_POINTER_PREFIX = b'version https://git-lfs.github.com/spec/'

ENV_WEIGHTS_DIR = 'WFI_SWIR_CGAN_WEIGHTS_DIR'
PACKAGE_WEIGHTS_DIR = Path(__file__).parent / 'weights'

# Non-tensor globals present in the training checkpoints (metric values saved
# as NumPy scalars). Allowlisting them lets us keep ``weights_only=True``.
_SAFE_GLOBALS = [
    (_np_scalar, 'numpy.core.multiarray.scalar'),
    _np_scalar,
    np.dtype,
    type(np.dtype('float64')),
    type(np.dtype('float32')),
]


def normalize_band(band):
    """Return the canonical band name (``'B11'`` or ``'B12'``)."""
    name = str(band).upper()
    aliases = {'SWIR1': 'B11', 'SWIR2': 'B12', '11': 'B11', '12': 'B12'}
    name = aliases.get(name, name)
    if name not in WEIGHT_FILES:
        raise ValueError(f"Unknown target band {band!r}. Use one of {sorted(WEIGHT_FILES)}.")
    return name


def _is_lfs_pointer(path):
    """Return True if ``path`` is a Git LFS pointer instead of the real file."""
    with open(path, 'rb') as f:
        return f.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX


def get_cache_dir():
    """Return the directory where downloaded weights are cached."""
    base = os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache'
    return Path(base) / 'wfi_swir_cgan'


def get_weights_path(band, download=True):
    """Return the local path of the checkpoint for ``band``, downloading it if needed."""
    filename = WEIGHT_FILES[normalize_band(band)]

    candidates = []
    if os.environ.get(ENV_WEIGHTS_DIR):
        candidates.append(Path(os.environ[ENV_WEIGHTS_DIR]) / filename)
    candidates.append(PACKAGE_WEIGHTS_DIR / filename)
    cached = get_cache_dir() / filename
    candidates.append(cached)

    for path in candidates:
        if path.is_file() and not _is_lfs_pointer(path):
            return path

    if not download:
        searched = '\n  '.join(str(p) for p in candidates)
        raise FileNotFoundError(f"Weights {filename} not found. Searched:\n  {searched}")

    url = f"{WEIGHTS_BASE_URL.rstrip('/')}/{filename}"
    print(f"[*] Downloading {filename} from {url}")
    cached.parent.mkdir(parents=True, exist_ok=True)
    tmp = cached.with_suffix('.part')
    try:
        torch.hub.download_url_to_file(url, str(tmp), progress=True)
        tmp.replace(cached)
    finally:
        if tmp.exists():
            tmp.unlink()
    return cached


def load_generator_state_dict(checkpoint_path):
    """Load the generator ``state_dict`` from a checkpoint file.

    Accepts full training checkpoints (``G_state_dict`` / ``generator_state_dict``
    keys) or a bare ``state_dict``. Loading uses ``weights_only=True``.
    """
    with torch.serialization.safe_globals(_SAFE_GLOBALS):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    if isinstance(checkpoint, dict) and 'G_state_dict' in checkpoint:
        state_dict = checkpoint['G_state_dict']
    elif isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
        state_dict = checkpoint['generator_state_dict']
    else:
        state_dict = checkpoint

    first_key = next(iter(state_dict))
    if first_key.startswith('module.'):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    return state_dict
