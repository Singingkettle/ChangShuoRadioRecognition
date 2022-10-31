# Copyright (c) Shuo Chang. All Rights Reserved.
import argparse
import os
import os.path as osp
import pickle
import zlib

import numpy as np
from tqdm import tqdm

from wtisp.common.fileio import load as IOLoad

_DATA_VERSION = ['201610A']


def parse_args():
    parser = argparse.ArgumentParser(description='Generate DeepSig Dataset')
    parser.add_argument('--data_root', type=str,
                        help='data root to deepsig data')
    args = parser.parse_args()

    return args


def run(dataset_folder, json_file):
    mode = json_file.split('.')[0]
    annotation = IOLoad(osp.join(dataset_folder, json_file))
    raw_data = {'sequence_data/ap': [],
                'sequence_data/iq': [],
                'constellation_data/filter_size_0.020_stride_0.020': [],
                'constellation_data/filter_size_0.050_stride_0.050': [],
                }
    compress_data = {'sequence_data/ap': [],
                     'sequence_data/iq': [],
                     'constellation_data/filter_size_0.020_stride_0.020': [],
                     'constellation_data/filter_size_0.050_stride_0.050': [],
                     }
    filename_index = dict()
    idx = 0
    for item_filename in tqdm(annotation['item_filename']):
        for key in raw_data:
            file_path = osp.join(dataset_folder, key, item_filename)
            data = np.load(file_path)
            data = data.astype(np.float64)
            raw_data[key].append(data)

            data = data.astype(np.float32)
            # it's not beneficial to compress sequence data
            cdata = zlib.compress(data.tobytes())
            compress_data[key].append(cdata)
        filename_index[item_filename] = idx
        idx += 1

    for key in raw_data:
        raw_data[key] = np.concatenate(raw_data[key], axis=0)
        compress_data[key] = b''.join(compress_data[key])

    mo_dir = osp.join(dataset_folder, 'memory_overhead')
    raw_dir = osp.join(dataset_folder, 'raw')
    if not osp.isdir(raw_dir):
        os.makedirs(raw_dir)

    if not osp.isdir(mo_dir):
        os.makedirs(mo_dir)

    for key in raw_data:
        file_name = key.split('/')[-1]

        com_file_path = osp.join(mo_dir, mode + '_' + file_name + '.pkl')
        pickle.dump(compress_data[key], open(com_file_path, 'wb'), protocol=4)

        raw_file_path = osp.join(raw_dir, mode + '_' + file_name + '.npy')
        pickle.dump(raw_data[key], open(raw_file_path, 'wb'), protocol=4)


def main():
    args = parse_args()
    for version in _DATA_VERSION:
        dataset_folder = osp.join(args.data_root, version)
        print(dataset_folder)
        # cache val data
        run(dataset_folder, 'validation.json')
        # cache test data
        run(dataset_folder, 'test.json')
        # cache train data
        run(dataset_folder, 'train.json')
        # cache train_and_val data
        run(dataset_folder, 'train_and_validation.json')


if __name__ == '__main__':
    main()
