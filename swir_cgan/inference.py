"""Tiled inference of SWIR bands (Sentinel-2-like B11/B12) from WFI imagery."""

import re
import warnings

import numpy as np
import torch
from scipy.signal.windows import tukey
from tqdm import tqdm

from .io import read_wfi, write_band
from .network import UNetGenerator
from .weights import get_weights_path, load_generator_state_dict, normalize_band

# Dropout rates used during training (irrelevant in eval mode, kept for fidelity).
DROPOUT = {'B11': 0.065, 'B12': 0.129}

REFLECTANCE_SCALE = 10000.0


def get_blend_window(patch_size, alpha=0.5):
    """
    Generates a 2D Tukey window: constant central plateau with smooth decay
    only at the overlapping edges, avoiding central brightness variations.
    """
    w1d = tukey(patch_size, alpha=alpha)
    w2d = np.outer(w1d, w1d)
    return np.clip(w2d, 1e-4, 1.0).astype(np.float32)


def cuda_device_supported(index=0):
    """Return True if the installed PyTorch has kernels for CUDA device ``index``.

    PyTorch wheels are compiled for a limited set of GPU architectures (e.g. the
    CUDA 13 builds require compute capability >= 7.5). Running on an older GPU fails
    with ``CUDA error: no kernel image is available for execution on the device``.
    """
    major, minor = torch.cuda.get_device_capability(index)
    for arch in torch.cuda.get_arch_list():
        match = re.fullmatch(r'(sm|compute)_(\d+)(\d)[a-z]?', arch)
        if not match:
            continue
        kind, arch_major, arch_minor = match.group(1), int(match.group(2)), int(match.group(3))
        # Binary code (sm) runs on the same major version with an equal or higher minor;
        # PTX (compute) can be JIT-compiled for any equal or newer architecture.
        if kind == 'sm' and arch_major == major and arch_minor <= minor:
            return True
        if kind == 'compute' and (arch_major, arch_minor) <= (major, minor):
            return True
    return False


def get_device(device=None):
    """Return a ``torch.device``.

    By default uses CUDA when it is available and supported by the installed
    PyTorch build, and falls back to the CPU otherwise.
    """
    if device is not None:
        return torch.device(device)
    if not torch.cuda.is_available():
        return torch.device('cpu')
    if not cuda_device_supported(0):
        major, minor = torch.cuda.get_device_capability(0)
        warnings.warn(
            f"GPU {torch.cuda.get_device_name(0)} (compute capability {major}.{minor}) is not "
            f"supported by the installed PyTorch {torch.__version__} (built for "
            f"{' '.join(torch.cuda.get_arch_list())}). Falling back to the CPU, which is much "
            "slower. To use the GPU, install a PyTorch build for an older CUDA version, e.g. "
            "`pip install torch --index-url https://download.pytorch.org/whl/cu126`.",
            stacklevel=2,
        )
        return torch.device('cpu')
    return torch.device('cuda')


def load_inference_model(band, weights_path=None, device=None, pad_input=16):
    """Build the generator for ``band`` ('B11' or 'B12') and load its weights."""
    band = normalize_band(band)
    device = get_device(device)
    checkpoint_path = weights_path or get_weights_path(band)
    print(f"[*] Loading model weights from: {checkpoint_path}")

    model = UNetGenerator(dropout_rate=DROPOUT[band], pad_input=pad_input).to(device)
    model.load_state_dict(load_generator_state_dict(checkpoint_path))
    model.eval()
    print(f"[+] Generator ({band}) loaded successfully on {device}.")
    return model


def predict_array(model, image, device, patch_size=128, overlap=64, batch_size=32):
    """Run tiled inference on a ``(4, H, W)`` reflectance array in ``[0, 1]``.

    Returns a ``(H, W)`` float32 array in ``[0, 1]``.
    """
    channels, height, width = image.shape

    # OMNIDIRECTIONAL PADDING (Protects the extremities)
    # Adds reflection on ALL sides. This way the extremities of the original image
    # will always fall in the CENTER of the U-Net's field of view.
    pad_t = overlap
    pad_l = overlap

    height_padded = height + pad_t
    width_padded = width + pad_l

    stride = patch_size - overlap
    pad_b = (stride - ((height_padded - patch_size) % stride)) % stride if height_padded >= patch_size else patch_size - height_padded
    pad_r = (stride - ((width_padded - patch_size) % stride)) % stride if width_padded >= patch_size else patch_size - width_padded

    image = np.pad(image, ((0, 0), (pad_t, pad_b), (pad_l, pad_r)), mode='reflect')

    proc_h, proc_w = image.shape[1], image.shape[2]

    output_image = np.zeros((proc_h, proc_w), dtype=np.float32)
    weight_map = np.zeros((proc_h, proc_w), dtype=np.float32)
    blend_weights = get_blend_window(patch_size, alpha=0.5)

    y_starts = list(range(0, proc_h - patch_size + 1, stride))
    x_starts = list(range(0, proc_w - patch_size + 1, stride))
    coords = [(y, x) for y in y_starts for x in x_starts]
    total_patches = len(coords)

    print(f"[*] Total patches: {total_patches} (Stride: {stride}px | Overlap: {overlap}px)")

    with torch.no_grad():
        for i in tqdm(range(0, total_patches, batch_size), desc="Batch Inference"):
            batch_coords = coords[i:i + batch_size]

            batch_patches = [
                image[:, y:y + patch_size, x:x + patch_size]
                for y, x in batch_coords
            ]

            batch_tensor = torch.from_numpy(np.stack(batch_patches, axis=0)).to(device)

            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                preds = model(batch_tensor)

            preds = preds.squeeze(1).float().cpu().numpy()

            for (y, x), pred_patch in zip(batch_coords, preds):
                output_image[y:y + patch_size, x:x + patch_size] += pred_patch * blend_weights
                weight_map[y:y + patch_size, x:x + patch_size] += blend_weights

    weight_map[weight_map == 0] = 1.0
    final_output = output_image / weight_map

    # CROP THE EXTRAPOLATED PADDING (Returns to exact original size)
    final_output = final_output[pad_t:pad_t + height, pad_l:pad_l + width]
    return np.clip(final_output, 0.0, 1.0)


class SWIRGenerator:
    """Reusable SWIR generator: loads the weights once and predicts many rasters.

    Parameters
    ----------
    band : str
        Target band: ``'B11'`` (SWIR1) or ``'B12'`` (SWIR2).
    weights_path : str | PathLike, optional
        Custom checkpoint. By default the packaged/cached weights are used and
        downloaded on first use.
    device : str | torch.device, optional
        ``'cuda'``, ``'cpu'``, ``'cuda:1'``... Defaults to CUDA when available.
    patch_size, overlap : int
        Tile size and overlap (in pixels) of the sliding window.
    batch_size : int
        Number of tiles per forward pass.
    """

    def __init__(self, band, weights_path=None, device=None,
                 patch_size=128, overlap=64, batch_size=32):
        self.band = normalize_band(band)
        self.device = get_device(device)
        self.patch_size = patch_size
        self.overlap = overlap
        self.batch_size = batch_size

        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        self.model = load_inference_model(self.band, weights_path, self.device)

    def predict(self, input_raster, output_raster=None):
        """Generate the SWIR band for ``input_raster``.

        Parameters
        ----------
        input_raster : str | PathLike | Sequence | Mapping
            A 4-band stack (blue, green, red, nir), a list of 4 single-band
            files in that order, or a dict with keys ``'blue'``, ``'green'``,
            ``'red'`` and ``'nir'``.
        output_raster : str | PathLike, optional
            Where to save the result as a GeoTIFF. If omitted, nothing is written.

        Returns
        -------
        np.ndarray
            The predicted band, ``(H, W)`` float32, in the same scale as the
            input (``[0, 1]`` or ``[0, 10000]``), with nodata restored.
        """
        print(f"[*] Processing input raster: {input_raster}")
        image, profile, invalid_mask, nodata = read_wfi(input_raster)
        valid_mask = ~invalid_mask

        # CRITICAL SANITIZATION (Avoids "Infection" of Patches at the edges)
        # Replaces backgrounds and NaNs with 0.0 BEFORE the neural network processes.
        # This ensures the U-Net does not return mathematical garbage in the boundary blocks.
        image[:, invalid_mask] = 0.0
        image = np.nan_to_num(image, nan=0.0)

        scaled_input = float(np.max(image)) > 1.0
        if scaled_input:
            print(f"[*] Values exceed 1.0. Adjusting factor {REFLECTANCE_SCALE} to [0, 1].")
            image = image / REFLECTANCE_SCALE
        image = np.clip(image, 0.0, 1.0)

        output = predict_array(self.model, image, self.device, self.patch_size,
                               self.overlap, self.batch_size)

        if scaled_input:
            output = output * REFLECTANCE_SCALE

        # FINAL TREATMENT: 3 Decimal Places and Perfect NoData Restoration
        output = np.round(output, 3).astype(np.float32)
        output[~valid_mask] = nodata

        if output_raster is not None:
            write_band(output_raster, output, profile, nodata)
            print(f"[+] Final raster successfully saved at: {output_raster}")
        return output


def generate_swir(band, input_raster, output_raster=None, batch_size=32, **kwargs):
    """One-shot helper: load the ``band`` model and predict a single input.

    Extra keyword arguments (``weights_path``, ``device``, ``patch_size``,
    ``overlap``) are forwarded to :class:`SWIRGenerator`.
    """
    generator = SWIRGenerator(band, batch_size=batch_size, **kwargs)
    return generator.predict(input_raster, output_raster)


def generate_swir1(input_raster, output_raster=None, batch_size=32, **kwargs):
    """Generate SWIR1 (Sentinel-2 B11, ~1610 nm). See :func:`generate_swir`."""
    return generate_swir('B11', input_raster, output_raster, batch_size, **kwargs)


def generate_swir2(input_raster, output_raster=None, batch_size=32, **kwargs):
    """Generate SWIR2 (Sentinel-2 B12, ~2190 nm). See :func:`generate_swir`."""
    return generate_swir('B12', input_raster, output_raster, batch_size, **kwargs)
