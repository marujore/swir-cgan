"""Locate, download and load the pretrained generator weights.

Weights are looked up in the following order:

1. An explicit path passed by the user (``weights_path=...``).
2. The directory given by the ``SWIR_CGAN_WEIGHTS_DIR`` environment variable.
3. The ``weights/`` directory inside the installed package (useful for a
   cloned repository or an editable install).
4. The user cache directory (``~/.cache/swir_cgan`` by default, or
   ``$XDG_CACHE_HOME/swir_cgan``). If the file is missing it is
   downloaded from the GitHub release of the selected version
   (``weights-v2`` by default, see ``WEIGHT_VERSIONS``) and its SHA256 is checked.

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

_RELEASES_URL = 'https://github.com/marujore/swir-cgan/releases/download'

# Published weight versions. Each one is a GitHub release (``weights-<version>``)
# with one checkpoint per band, stored as ``(file name, SHA256)``. Downloads whose
# SHA256 does not match are rejected. The ``.pth`` files are not shipped in the
# wheel (PyPI limits files to 100 MB). To publish new weights, create a new release
# (``weights-v3``...), add it here and update ``DEFAULT_WEIGHTS_VERSION``.
WEIGHT_VERSIONS = {
    'v1': {  # ~200 MB per band, best epoch of the original training.
        'B11': ('best_model_B11.pth',
                'ca82911a8e4668466099eae51ce2e525fe82bbbd2b04cf46b7325cf1088e676c'),
        'B12': ('best_model_B12.pth',
                '9655ccdf590a281af3fb8705246ac9ba6f1f27af119c3b780085b7315155ed5b'),
    },
    'v2': {  # ~600 MB per band (full training checkpoints).
        'B11': ('B11_pretrained_weights_v2.pth',
                '4e456912362ecbf230ecb847eb9485df72355e31cc41d3d39bc6e0989733831e'),
        'B12': ('B12_pretrained_weights_v2.pth',
                'dc7977626912e7663b57d308cfa2f810382aab4cac04408b07be4ae716cf0a49'),
    },
}
DEFAULT_WEIGHTS_VERSION = 'v2'
BANDS = ('B11', 'B12')

# Optional mirror replacing the GitHub release URL of the selected version. It must
# serve the same file names.
ENV_WEIGHTS_URL = 'SWIR_CGAN_WEIGHTS_URL'

_LFS_POINTER_PREFIX = b'version https://git-lfs.github.com/spec/'

ENV_WEIGHTS_DIR = 'SWIR_CGAN_WEIGHTS_DIR'
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
    if name not in BANDS:
        raise ValueError(f"Unknown target band {band!r}. Use one of {list(BANDS)}.")
    return name


def normalize_version(version=None):
    """Return the canonical weights version (``'v1'``, ``'v2'``...); ``None`` means the default."""
    if version is None:
        return DEFAULT_WEIGHTS_VERSION
    name = str(version).lower().removeprefix('weights-')
    if not name.startswith('v'):
        name = f'v{name}'
    if name not in WEIGHT_VERSIONS:
        raise ValueError(f"Unknown weights version {version!r}. "
                         f"Use one of {list(WEIGHT_VERSIONS)}.")
    return name


def weights_filename(band, version=None):
    """Return the checkpoint file name for ``band`` in weights ``version``."""
    return WEIGHT_VERSIONS[normalize_version(version)][normalize_band(band)][0]


def _is_lfs_pointer(path):
    """Return True if ``path`` is a Git LFS pointer instead of the real file."""
    with open(path, 'rb') as f:
        return f.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX


def get_cache_dir():
    """Return the directory where downloaded weights are cached."""
    base = os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache'
    return Path(base) / 'swir_cgan'


def get_weights_path(band, version=None, download=True):
    """Return the local path of the checkpoint for ``band``, downloading it if needed.

    ``version`` selects the weights release (``'v1'``, ``'v2'``...). Defaults to
    ``DEFAULT_WEIGHTS_VERSION``.
    """
    version = normalize_version(version)
    filename, sha256 = WEIGHT_VERSIONS[version][normalize_band(band)]

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

    base_url = os.environ.get(ENV_WEIGHTS_URL) or f'{_RELEASES_URL}/weights-{version}'
    url = f"{base_url.rstrip('/')}/{filename}"
    print(f"[*] Downloading {filename} from {url}")
    cached.parent.mkdir(parents=True, exist_ok=True)
    tmp = cached.with_suffix('.part')
    try:
        # Raises RuntimeError if the SHA256 of the downloaded file does not match.
        torch.hub.download_url_to_file(url, str(tmp), hash_prefix=sha256,
                                       progress=True)
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
