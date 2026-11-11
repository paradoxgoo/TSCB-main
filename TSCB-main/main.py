"""Run the IEEE-CIS experiments in Section 5.2."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    from tscb.realistic_delay import run

    run(args.data_dir, args.output / f'seed-{args.seed}', args.seed, args.cpu)


if __name__ == '__main__':
    main()
