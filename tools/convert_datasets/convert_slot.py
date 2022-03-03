# Copyright (c) Shuo Chang. All Rights Reserved.

import argparse
import glob
import os
from wtislot.slot_class import Slot, RawIQ

# _MAT_PREFIX = [
#     'IQ_128_Ds10_36000', 'IQ_128_Ds150_36000',
#     'Con_128_Ds10_36000', 'Con_128_Ds10_4000', 'Con_128_Ds150_1200', 'Con_128_Ds150_36000',
#     'Con_64_Ds10_36000', 'Con_64_Ds10_4000', 'Con_64_Ds150_1200', 'Con_64_Ds150_36000',
#     'Con_32_Ds10_36000', 'Con_32_Ds10_4000',
# ]

_MAT_PREFIX = [
    # 'IQ_128_Ds10_36000',
    'Con_128_Ds10_36000',
]


def parse_args():
    parser = argparse.ArgumentParser(description='Convert ZRYSlot Dataset')
    parser.add_argument('--data_root', type=str,
                        default='/home/citybuster/Data/SignalProcessing/ModulationClassification',
                        help='data root to save converted data')
    args = parser.parse_args()

    return args


def generate(data_generator, mat_file_path):
    data_generator.set_mode('train')
    data_generator.set_file_path(mat_file_path)
    data_generator.generate()

    data_generator.set_mode('test')
    data_generator.set_file_path(mat_file_path)
    data_generator.generate()


def main():
    raw_mat_dir = '/home/citybuster/Data/Rayleighsnr'
    args = parse_args()
    data_root = args.data_root
    for prefix_str in _MAT_PREFIX:
        if 'Con' in prefix_str:
            data_generator = Slot(data_root, prefix_str)
        else:
            data_generator = RawIQ(data_root, prefix_str)

        # Process train mat file
        if 'Con' in prefix_str:
            mat_file_path = glob.glob(os.path.join(
                raw_mat_dir, 'strain', 'Cons', prefix_str + '*.mat'))
        else:
            mat_file_path = glob.glob(os.path.join(
                raw_mat_dir, 'strain', 'IQ_two', prefix_str + '*.mat'))
        data_generator.set_mode('train')
        data_generator.set_file_path(mat_file_path[0])
        data_generator.generate()

        # Process test mat files
        if 'Con' in prefix_str:
            mat_file_paths = glob.glob(os.path.join(
                raw_mat_dir, 'stest_paper', 'Cons', prefix_str + '*.mat'))
        else:
            mat_file_paths = glob.glob(os.path.join(
                raw_mat_dir, 'stest_paper', 'IQ_two', prefix_str + '*.mat'))
        data_generator.set_mode('test')
        data_generator.set_file_path(mat_file_paths)
        data_generator.generate()


if __name__ == '__main__':
    main()
