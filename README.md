# swir-cgan

A Short-Wave Infrared (SWIR) synthesis framework for the Wide Field Imager (WFI) sensor.

`swir-cgan` generates synthetic SWIR bands for images from the WFI cameras on board
**CBERS-4, CBERS-4A and AMAZONIA-1**. These cameras have no SWIR bands. From the four WFI bands
(**blue, green, red and near-infrared (NIR)**), the package generates the two Sentinel-2-like
SWIR bands: **B11 (SWIR1, ~1610 nm)** and **B12 (SWIR2, ~2190 nm)**.
It uses a conditional GAN with an attention U-Net generator.

> **Input images must be Top of Atmosphere (ToA) reflectance.**

## Installation

```bash
pip install git+https://github.com/marujore/swir-cgan.git
```

or from a local clone:

```bash
git clone https://github.com/marujore/swir-cgan.git
cd swir-cgan
pip install -e .
```

or, with conda, from a local clone (this installs rasterio from conda-forge):

```bash
conda env create -f environment.yml
conda activate swir-cgan
```

Requirements: Python ≥ 3.10, PyTorch ≥ 2.4, rasterio, NumPy, SciPy and tqdm.
A CUDA-capable GPU is recommended but not required.
To get a specific CUDA build of PyTorch, install it first by following the
[PyTorch instructions](https://pytorch.org/get-started/locally/).

**Older GPUs.** The default PyTorch wheels on PyPI are built for recent GPUs only. Recent
versions use CUDA 13, which requires compute capability ≥ 7.5. On older GPUs, such as Pascal or
Volta, the package warns and runs on the CPU. To use these GPUs, install PyTorch built for
CUDA 12.6:

```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
```

Check the result with
`python -c "import torch; print(torch.cuda.get_arch_list(), torch.cuda.get_device_capability())"`.

## Pretrained weights

There is one generator per band: `best_model_B11.pth` and `best_model_B12.pth`, about 200 MB each.
They are published as assets of the
[`weights-v1` release](https://github.com/marujore/swir-cgan/releases/tag/weights-v1).
They are too large to ship inside the Python package, so the package **downloads them
automatically the first time they are needed** and caches them in `~/.cache/swir_cgan/`.
Only the band you use is downloaded.

The package looks for the weights in this order:

1. The `weights_path=` argument (Python) or `--weights-dir` (CLI).
2. The directory in the `SWIR_CGAN_WEIGHTS_DIR` environment variable.
3. `swir_cgan/weights/` inside the package, if you place the files there (for example
   in a clone installed with `pip install -e .`). The `.pth` files there are ignored by Git.
4. The cache directory (`$XDG_CACHE_HOME/swir_cgan` or `~/.cache/swir_cgan`).
   If the file is missing there, it is downloaded from the `weights-v1` release.

For offline machines, copy the two `.pth` files to a folder and set
`export SWIR_CGAN_WEIGHTS_DIR=/path/to/folder`.

## Quick start

### 4-band stack

```python
from swir_cgan import generate_swir1, generate_swir2

generate_swir1('wfi_stack.tif', 'wfi_B11.tif')
generate_swir2('wfi_stack.tif', 'wfi_B12.tif', batch_size=16)
```

The bands in the stack must be in the order **blue, green, red, nir**.

### Four separate files

Pass a list in the order `[blue, green, red, nir]`:

```python
generate_swir1(
    ['wfi_blue.tif', 'wfi_green.tif', 'wfi_red.tif', 'wfi_nir.tif'],
    'wfi_B11.tif',
)
```

or a dictionary, where the order does not matter:

```python
bands = {
    'blue': 'wfi_blue.tif',
    'green': 'wfi_green.tif',
    'red': 'wfi_red.tif',
    'nir': 'wfi_nir.tif',
}
generate_swir2(bands, 'wfi_B12.tif')
```

The four files must share the same size, geotransform and CRS. If they don't, a `ValueError` is raised.

### Processing many scenes

`generate_swir1` and `generate_swir2` load the model on every call. To process several scenes,
create a `SWIRGenerator` once and reuse it:

```python
from swir_cgan import SWIRGenerator

generator = SWIRGenerator('B11', device='cuda', batch_size=32)
for scene in ['scene_a.tif', 'scene_b.tif']:
    generator.predict(scene, f'B11_{scene}')
```

`predict` also returns the result as a NumPy array. If `output_raster` is omitted,
nothing is written to disk.

More examples are in [`examples/example.py`](examples/example.py).

### Command line

```bash
# From a stack, generating both bands
swir-cgan --stack wfi_stack.tif --b11 out_B11.tif --b12 out_B12.tif

# From separate files (order: blue green red nir)
swir-cgan --bands blue.tif green.tif red.tif nir.tif --b11 out_B11.tif --device cpu
```

Run `swir-cgan --help` for all options.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `input_raster` | — | Path to a 4-band stack, a list of 4 band files, or a dict with the keys `blue`, `green`, `red`, `nir`. |
| `output_raster` | `None` | Output GeoTIFF path. If `None`, only the array is returned. |
| `batch_size` | `32` | Number of 128×128 tiles per forward pass. Reduce it if the GPU runs out of memory. |
| `device` | auto | `'cuda'`, `'cuda:1'`, `'cpu'`… Defaults to CUDA when available. |
| `weights_path` | auto | Custom checkpoint file. |
| `patch_size` / `overlap` | `128` / `64` | Size and overlap of the sliding-window tiles, in pixels. |

## Input and output conventions

- **Processing level.** The WFI bands (CBERS-4, CBERS-4A or AMAZONIA-1) must be
  **Top of Atmosphere (ToA) reflectance**.
- **Reflectance scale.** Inputs can be in `[0, 1]` or scaled by 10,000 (`[0, 10000]`). Scaling is
  detected automatically: if any value is above 1, the image is divided by 10,000. The output uses
  the same scale as the input.
- **Nodata.** A pixel is treated as nodata when *all* bands equal their nodata value, or when any
  band is NaN. These pixels are filled with 0 before inference and set back to nodata in the output.
  The nodata value comes from the input files (default `-9999`).
- **Output.** A single-band, LZW-compressed, float32 GeoTIFF on the same grid as the input, with
  values rounded to 3 decimals.
- **Memory.** The whole scene is loaded into memory.

## How it works

1. The four WFI bands are normalized to `[0, 1]` and reflection-padded on every side, so pixels at
   the image edges are also predicted near the center of a tile.
2. The image is split into overlapping 128×128 tiles (stride 64) and processed in batches by the
   attention U-Net generator (`swir_cgan.network.UNetGenerator`).
3. Tile predictions are combined with a 2D Tukey window. This removes seams between tiles without
   changing brightness in the tile centers.
4. The padding is cropped, the original scale is restored and the nodata mask is reapplied.

### Metrics

Metrics stored in each checkpoint for the best training epoch (reflectance in `[0, 1]`):

| Band | SSIM | PSNR (dB) | R² | MAE | RMSE |
|---|---|---|---|---|---|
| B11 (SWIR1) | 0.905 | 31.86 | 0.920 | 0.0186 | 0.0255 |
| B12 (SWIR2) | 0.920 | 33.68 | 0.910 | 0.0145 | 0.0207 |

## Project layout

```
swir_cgan/
├── __init__.py      # public API
├── inference.py     # SWIRGenerator, generate_swir1/2, tiled inference
├── io.py            # reading stacks or separate band files, writing GeoTIFFs
├── network.py       # UNetGenerator and PatchDiscriminator
├── weights.py       # locating / downloading / loading checkpoints
├── cli.py           # `swir-cgan` command
└── weights/         # optional local copy of the checkpoints (not tracked)
examples/            # usage examples
scripts/             # maintenance scripts (weights export)
tests/               # pytest suite
```

## Citation

If you use this package in your research, please cite:

```
TODO: add the paper reference / BibTeX here.
```

## License

Distributed under the GNU General Public License v3.0. See [LICENSE](LICENSE).
