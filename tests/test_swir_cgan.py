import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from swir_cgan.inference import predict_array
from swir_cgan.io import BAND_ORDER, read_wfi, write_band

H, W = 50, 70
TRANSFORM = from_origin(500000, 8000000, 64, 64)


def _write(path, data, nodata=0.0, transform=TRANSFORM):
    if data.ndim == 2:
        data = data[None]
    with rasterio.open(path, 'w', driver='GTiff', width=data.shape[-1], height=data.shape[-2],
                       count=data.shape[0], dtype='float32', crs='EPSG:32723',
                       transform=transform, nodata=nodata) as dst:
        dst.write(data.astype('float32'))
    return path


@pytest.fixture
def image():
    rng = np.random.default_rng(0)
    img = rng.uniform(100, 5000, size=(4, H, W)).astype('float32')
    img[:, :5, :5] = 0.0  # nodata corner
    return img


def test_stack_and_separate_files_match(tmp_path, image):
    stack = _write(tmp_path / 'stack.tif', image)
    paths = [_write(tmp_path / f'{b}.tif', image[i][None]) for i, b in enumerate(BAND_ORDER)]

    a, meta_a, mask_a, nd_a = read_wfi(stack)
    b, meta_b, mask_b, nd_b = read_wfi(paths)
    c, _, mask_c, _ = read_wfi(dict(zip(BAND_ORDER, paths)))

    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(mask_a, mask_b)
    assert mask_a[:5, :5].all() and mask_a.sum() == 25
    assert nd_a == nd_b == 0.0
    assert meta_a['transform'] == meta_b['transform']


def test_dict_keys_are_validated(tmp_path, image):
    paths = [_write(tmp_path / f'{b}.tif', image[i][None]) for i, b in enumerate(BAND_ORDER)]
    with pytest.raises(ValueError, match='Missing'):
        read_wfi({'blue': paths[0], 'green': paths[1], 'red': paths[2]})


def test_wrong_number_of_bands(tmp_path, image):
    with pytest.raises(ValueError, match='expects 4'):
        read_wfi(_write(tmp_path / 'three.tif', image[:3]))
    with pytest.raises(ValueError, match='Expected 4'):
        read_wfi([tmp_path / 'a.tif'] * 3)


def test_mismatched_grid_is_rejected(tmp_path, image):
    paths = [_write(tmp_path / f'{b}.tif', image[i][None]) for i, b in enumerate(BAND_ORDER)]
    paths[3] = _write(tmp_path / 'nir_shifted.tif', image[3][None], transform=from_origin(0, 0, 64, 64))
    with pytest.raises(ValueError, match='grid'):
        read_wfi(paths)


def test_predict_array_preserves_shape(image):
    class Mean(torch.nn.Module):
        def forward(self, x):
            return x.mean(dim=1, keepdim=True)

    img = np.clip(image / 10000.0, 0, 1)
    out = predict_array(Mean(), img, torch.device('cpu'), patch_size=32, overlap=16, batch_size=8)
    assert out.shape == (H, W)
    # Tiled blending of a pixel-wise operator must reproduce it exactly.
    np.testing.assert_allclose(out, img.mean(axis=0), atol=1e-5)


def test_write_band_in_current_dir(tmp_path, monkeypatch, image):
    monkeypatch.chdir(tmp_path)
    _, meta, _, nodata = read_wfi(_write(tmp_path / 'stack.tif', image))
    write_band('out.tif', image[0], meta, nodata)
    with rasterio.open('out.tif') as src:
        assert src.count == 1 and src.nodata == nodata
