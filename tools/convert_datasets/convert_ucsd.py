# Copyright (c) Shuo Chang. All Rights Reserved.

import argparse

from datasets import UCSDRML22

_DATA_VERSION = ['RML22']

_UCSD_CLASSES = {
    'RML22': UCSDRML22,
}


def build_deepsig_class(version, data_root, data_ratios):
    deepsig_class = _UCSD_CLASSES[version]
    return deepsig_class(data_root, version, data_ratios)


def parse_args():
    parser = argparse.ArgumentParser(description='Convert UCSD Dataset')
    parser.add_argument('--data_root', type=str,
                        help='data root to csrr data')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    data_ratios = [0.5, 0.1, 0.4]
    for version in _DATA_VERSION:
        data_generator = build_deepsig_class(
            version, args.data_root, data_ratios)
        data_generator.generate()


if __name__ == '__main__':
    main()
