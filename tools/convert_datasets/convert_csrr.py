# Copyright (c) Shuo Chang. All Rights Reserved.

import argparse

from datasets import CSRRData


def parse_args():
    parser = argparse.ArgumentParser(description='Convert CSRR Dataset')
    parser.add_argument('--data_root', type=str,
                        help='data root to csrr data')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    data_ratios = [0.6, 0.2, 0.2]
    for version in range(1, 42):
        data_generator = CSRRData(args.data_root, f'v{version}', data_ratios)
        data_generator.generate()


if __name__ == '__main__':
    main()
