"""Export generator-only checkpoints, ready to be committed to the repository (Git LFS).

Training checkpoints also store the discriminator and NumPy metric values.
This script keeps only what inference needs (generator weights, training
hyperparameters and metrics as plain Python floats).

Usage:
    python scripts/export_weights.py path/to/best_model_B11.pth out/best_model_B11.pth
"""

import argparse

import torch

from wfi_swir_cgan.weights import _SAFE_GLOBALS


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('checkpoint')
    parser.add_argument('output')
    args = parser.parse_args()

    with torch.serialization.safe_globals(_SAFE_GLOBALS):
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)

    exported = {
        'G_state_dict': ckpt['G_state_dict'],
        'epoch': int(ckpt.get('epoch', -1)),
        'hyperparameters': ckpt.get('hyperparameters', {}),
        'metrics': {k.removeprefix('best_'): float(v) for k, v in ckpt.items() if k.startswith('best_')},
    }
    torch.save(exported, args.output)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
