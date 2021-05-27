# Copyright (c) Shuo Chang. All Rights Reserved.

import argparse

from deepsig.data_class import DeepSigA, DeepSigB, DeepSigC, DeepSigD

# _DATA_VERSION = ['201604C', '201610A', '201610B', '201801A']
_DATA_VERSION = ['201801A']

_DEEPSIG_CLASSES = {
    '201604C': DeepSigA,
    '201610A': DeepSigB,
    '201610B': DeepSigC,
    '201801A': DeepSigD,
}


def build_deepsig_class(version, data_root, data_ratios):
    deepsig_class = _DEEPSIG_CLASSES[version]
    return deepsig_class(data_root, version, data_ratios)


def parse_args():
    parser = argparse.ArgumentParser(description='Convert DeepSig Dataset')
    parser.add_argument('--data_root', type=str,
                        help='data root to deepsig data')
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
