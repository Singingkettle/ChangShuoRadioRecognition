# Copyright (c) Shuo Chang. All Rights Reserved.
import argparse
import os
import os.path as osp
import pickle
import zlib

import numpy as np
from tqdm import tqdm

from csrr.common.fileio import load as IOLoad

# _DATA_VERSION = ['201801A', '201610A', '201604C', '201610B']

_DATA_VERSION = ['201610A', '201801A']


def parse_args():
    parser = argparse.ArgumentParser(description='Generate DeepSig Dataset')
    parser.add_argument('--data_root', type=str,
                        help='data root to deepsig data')
    args = parser.parse_args()

    return args


def run(dataset_folder, json_file):
    mode = json_file.split('.')[0]
    annotations = IOLoad(osp.join(dataset_folder, json_file))
    cache_data = {'sequence_data/ap': {'data': [], 'shape': None, 'dtype': None},
                  'sequence_data/iq': {'data': [], 'shape': None, 'dtype': None},
                  'constellation_data/filter_size_0.020_stride_0.020': {'data': [], 'shape': None, 'dtype': None},
                  'constellation_data/filter_size_0.050_stride_0.050': {'data': [], 'shape': None, 'dtype': None},
                  }
    file_name_index = dict()
    idx = 0
    for annotation in tqdm(annotations['annotations']):
        file_name = annotation['file_name']
        for key in cache_data:
            file_path = osp.join(dataset_folder, key, file_name)
            data = np.load(file_path)
            data = data.astype(np.float32)
            cache_data[key]['shape'] = data.shape
            cache_data[key]['dtype'] = data.dtype
            # it's not beneficial to compress sequence data
            if 'sequence_data' in key:
                data = np.expand_dims(data, axis=0)
                cache_data[key]['data'].append(data)
            else:
                cdata = zlib.compress(data.tobytes())
                cache_data[key]['data'].append(cdata)
        file_name_index[file_name] = idx
        idx += 1

    for key in cache_data:
        if 'sequence_data' in key:
            cache_data[key]['data'] = np.concatenate(cache_data[key]['data'], axis=0)
        cache_data[key]['lookup_table'] = file_name_index

    cache_dir = osp.join(dataset_folder, 'cache')
    if not osp.isdir(cache_dir):
        os.makedirs(cache_dir)

    for key in cache_data:
        cache_name = key.split('/')[-1]
        cache_file_path = osp.join(cache_dir, mode + '_' + cache_name + '.pkl')
        pickle.dump(cache_data[key], open(cache_file_path, 'wb'), protocol=4)


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
