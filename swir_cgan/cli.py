"""Command line interface: ``swir-cgan``."""

import argparse
from pathlib import Path

from .inference import SWIRGenerator
from .io import BAND_ORDER
from .weights import DEFAULT_WEIGHTS_VERSION, WEIGHT_VERSIONS, weights_filename


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='swir-cgan',
        description='Generate SWIR bands (B11/B12) from WFI blue, green, red and NIR bands.',
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--stack', help='4-band raster stacked as blue, green, red, nir.')
    src.add_argument('--bands', nargs=4, metavar=tuple(b.upper() for b in BAND_ORDER),
                     help='4 single-band rasters, in the order blue green red nir.')
    parser.add_argument('--b11', help='Output path for SWIR1 (B11).')
    parser.add_argument('--b12', help='Output path for SWIR2 (B12).')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default=None, help="e.g. 'cuda', 'cpu'. Default: CUDA if available.")
    parser.add_argument('--weights-version', choices=list(WEIGHT_VERSIONS),
                        default=DEFAULT_WEIGHTS_VERSION,
                        help='Published weights to use (default: %(default)s).')
    parser.add_argument('--weights-dir', default=None,
                        help='Directory containing the checkpoints of the selected '
                             "--weights-version (e.g. B11_pretrained_weights_v2.pth for v2).")
    args = parser.parse_args(argv)

    if not (args.b11 or args.b12):
        parser.error('at least one of --b11 or --b12 is required.')

    input_raster = args.stack or args.bands
    for band, output in (('B11', args.b11), ('B12', args.b12)):
        if not output:
            continue
        weights_path = None
        if args.weights_dir:
            weights_path = Path(args.weights_dir) / weights_filename(band, args.weights_version)
        generator = SWIRGenerator(band, weights_path=weights_path, device=args.device,
                                  batch_size=args.batch_size,
                                  weights_version=args.weights_version)
        generator.predict(input_raster, output)


if __name__ == '__main__':
    main()
