#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Project: wtisignalprocessing
File: convert_matlab.py
Author: Citybuster
Time: 2021/7/5 14:07
Email: chagshuo@bupt.edu.cn
"""

import argparse

from matlab.data_class import MatlabData

# _DATA_VERSION = ['201604C', '201610A', '201610B', '201801A']
_DATA_VERSION = ['202107A']


def build_matlab_class(version, data_root, data_ratios):
    return MatlabData(data_root, version, data_ratios)


def parse_args():
    parser = argparse.ArgumentParser(description='Convert Matlab Dataset')
    parser.add_argument('--data_root', type=str,
                        help='data root to matlab data')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    data_ratios = [0.5, 0.1, 0.4]
    for version in _DATA_VERSION:
        data_generator = build_matlab_class(
            version, args.data_root, data_ratios)
        data_generator.generate()


if __name__ == '__main__':
    main()
