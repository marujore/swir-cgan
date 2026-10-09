"""WFI-SWIR-cGAN: synthesis of SWIR bands for the Wide Field Imager (WFI) sensor."""

from .inference import (
    SWIRGenerator,
    generate_swir,
    generate_swir1,
    generate_swir2,
    load_inference_model,
)
from .io import BAND_ORDER, read_wfi
from .weights import get_weights_path

__version__ = '0.1.0'

__all__ = [
    'BAND_ORDER',
    'SWIRGenerator',
    'generate_swir',
    'generate_swir1',
    'generate_swir2',
    'get_weights_path',
    'load_inference_model',
    'read_wfi',
    '__version__',
]
