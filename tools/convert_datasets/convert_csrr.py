# Copyright (c) Shuo Chang. All Rights Reserved.

import argparse

from datasets import CSRRData

_DATA_VERSION = ['CSRR2023']

_CSRR_CLASSES = {
    'CSRR2023': CSRRData,
}


def build_csrr_class(version, data_root, data_ratios):
    csrr_class = _CSRR_CLASSES[version]
    return csrr_class(data_root, version, data_ratios)


def parse_args():
    parser = argparse.ArgumentParser(description='Convert CSRR Dataset')
    parser.add_argument('--data_root', type=str,
                        help='data root to csrr data')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    data_ratios = [0.6, 0.2, 0.2]
    for version in _DATA_VERSION:
        data_generator = build_csrr_class(
            version, args.data_root, data_ratios)
        data_generator.generate()


if __name__ == '__main__':
    main()
