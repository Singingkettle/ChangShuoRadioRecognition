# Copyright (c) Shuo Chang. All Rights Reserved.

import argparse

from wtiss.data_class import ISSDataBase

_DATA_VERSION = ['qpsk_16qam']

_DATA_DB = ['20dB', '5dB']

_DATA_MODE = ['complex', 'real']


def parse_args():
    parser = argparse.ArgumentParser(description='Convert DeepSig Dataset')
    parser.add_argument('--data_root', type=str,
                        help='data root to deepsig data',
                        default='/home/citybuster/Data/wtisignalprocessing/SignalSeparation')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    data_ratios = [0.5, 0.1, 0.4]
    for version in _DATA_VERSION:
        for dB in _DATA_DB:
            for mode in _DATA_MODE:
                data_processor = ISSDataBase(
                    args.data_root, version, dB, mode, data_ratios)
                data_processor.process_data()


if __name__ == '__main__':
    main()
