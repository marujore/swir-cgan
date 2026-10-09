"""Reading WFI inputs either as a single 4-band stack or as 4 single-band files."""

import os
from collections.abc import Mapping

import numpy as np
import rasterio

# Band order expected by the generator (channel 0 to 3).
BAND_ORDER = ('blue', 'green', 'red', 'nir')

DEFAULT_NODATA = -9999.0


def _band_paths(input_raster):
    """Normalize the user input into either one stack path or 4 ordered band paths.

    Returns ``(stack_path, None)`` or ``(None, [blue, green, red, nir])``.
    """
    if isinstance(input_raster, (str, os.PathLike)):
        return input_raster, None

    if isinstance(input_raster, Mapping):
        keys = {str(k).lower(): v for k, v in input_raster.items()}
        missing = [b for b in BAND_ORDER if b not in keys]
        extra = sorted(set(keys) - set(BAND_ORDER))
        if missing or extra:
            raise ValueError(
                f"Band dictionary must have exactly the keys {BAND_ORDER}. "
                f"Missing: {missing}. Unexpected: {extra}."
            )
        return None, [keys[b] for b in BAND_ORDER]

    paths = list(input_raster)
    if len(paths) != len(BAND_ORDER):
        raise ValueError(
            f"Expected {len(BAND_ORDER)} band files ordered as {BAND_ORDER}, got {len(paths)}."
        )
    return None, paths


def read_wfi(input_raster):
    """Read the WFI input into memory.

    Parameters
    ----------
    input_raster : str | PathLike | Sequence | Mapping
        * Path to a 4-band raster stacked as ``(blue, green, red, nir)``;
        * a list/tuple with 4 single-band rasters in the order
          ``[blue, green, red, nir]``;
        * a dict with the keys ``'blue'``, ``'green'``, ``'red'`` and ``'nir'``.

    Returns
    -------
    image : np.ndarray
        Float32 array with shape ``(4, H, W)``.
    profile : dict
        Rasterio metadata (``src.meta``) of the (first) input raster.
    invalid_mask : np.ndarray
        Boolean ``(H, W)`` mask; ``True`` where all bands are nodata or any band is NaN.
    nodata : float
        Nodata value used for the output.
    """
    stack_path, band_paths = _band_paths(input_raster)

    if stack_path is not None:
        with rasterio.open(stack_path) as src:
            profile = src.meta.copy()
            image = src.read().astype(np.float32)
            band_nodata = [src.nodata] * src.count
        if image.shape[0] != len(BAND_ORDER):
            raise ValueError(
                f"The model expects {len(BAND_ORDER)} WFI bands {BAND_ORDER}, "
                f"but {stack_path} has {image.shape[0]}."
            )
    else:
        arrays, band_nodata, profile = [], [], None
        for name, path in zip(BAND_ORDER, band_paths):
            with rasterio.open(path) as src:
                if src.count != 1:
                    raise ValueError(f"Band file for {name!r} ({path}) must have 1 band, found {src.count}.")
                if profile is None:
                    profile = src.meta.copy()
                elif (src.width, src.height) != (profile['width'], profile['height']) \
                        or src.transform != profile['transform'] or src.crs != profile['crs']:
                    raise ValueError(
                        f"Band file for {name!r} ({path}) does not share the grid "
                        f"(size, transform, CRS) of the {BAND_ORDER[0]!r} file."
                    )
                arrays.append(src.read(1).astype(np.float32))
                band_nodata.append(src.nodata)
        image = np.stack(arrays, axis=0)

    nodata = next((v for v in band_nodata if v is not None), DEFAULT_NODATA)
    band_nodata = [nodata if v is None else v for v in band_nodata]

    is_nodata = np.stack([band == nd for band, nd in zip(image, band_nodata)], axis=0)
    invalid_mask = np.all(is_nodata, axis=0) | np.any(np.isnan(image), axis=0)
    return image, profile, invalid_mask, float(nodata)


def write_band(output_raster, array, profile, nodata):
    """Write a single-band float32 GeoTIFF using the grid of ``profile``."""
    profile = profile.copy()
    profile.update(driver='GTiff', count=1, dtype='float32', compress='lzw', nodata=nodata)

    out_dir = os.path.dirname(os.fspath(output_raster))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with rasterio.open(output_raster, 'w', **profile) as dst:
        dst.write(array.astype(np.float32), 1)
