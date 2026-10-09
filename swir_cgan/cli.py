"""Command line interface: ``swir-cgan``."""

import argparse

from .inference import SWIRGenerator
from .io import BAND_ORDER


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
    parser.add_argument('--weights-dir', default=None,
                        help='Directory containing best_model_B11.pth / best_model_B12.pth.')
    args = parser.parse_args(argv)

    if not (args.b11 or args.b12):
        parser.error('at least one of --b11 or --b12 is required.')

    input_raster = args.stack or args.bands
    for band, output in (('B11', args.b11), ('B12', args.b12)):
        if not output:
            continue
        weights_path = f'{args.weights_dir}/best_model_{band}.pth' if args.weights_dir else None
        generator = SWIRGenerator(band, weights_path=weights_path, device=args.device,
                                  batch_size=args.batch_size)
        generator.predict(input_raster, output)


if __name__ == '__main__':
    main()
