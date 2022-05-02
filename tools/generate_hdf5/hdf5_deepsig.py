# Copyright (c) Shuo Chang. All Rights Reserved.
import argparse
import os
import os.path as osp
import pickle
import zlib

import h5py
import numpy as np

_DATA_VERSION = ['201801A', '201610A', '201604C', '201610B']


def parse_args():
    parser = argparse.ArgumentParser(description='Generate hdf5 of DeepSig Dataset')
    parser.add_argument('--data_root', type=str, help='data root to deepsig data')
    args = parser.parse_args()

    return args


def run(dataset_folder, json_file):
    mode = json_file.split('.')[0]
    cache_data = {'sequence_data/ap': {'data': [], 'shape': None, 'dtype': None},
                  'sequence_data/iq': {'data': [], 'shape': None, 'dtype': None},
                  'constellation_data/filter_size_0.020_stride_0.020': {'data': [], 'shape': None, 'dtype': None},
                  'constellation_data/filter_size_0.050_stride_0.050': {'data': [], 'shape': None, 'dtype': None},
                  }

    cache_dir = osp.join(dataset_folder, 'cache')
    hdf5_dir = osp.join(dataset_folder, 'hdf5')
    if not osp.isdir(hdf5_dir):
        os.makedirs(hdf5_dir)

    for key in cache_data:
        cache_name = key.split('/')[-1]
        cache_file_path = osp.join(cache_dir, mode + '_' + cache_name + '.pkl')
        hdf5_file_path = osp.join(hdf5_dir, mode + '_' + cache_name + '.h5')
        look_table_file_path = osp.join(hdf5_dir, mode + '_' + cache_name + '.pkl')
        data = pickle.load(open(osp.join(cache_file_path), 'rb'))
        if 'filter' in cache_name:
            continue
            # items = [zlib.decompress(x) for x in data['data']]
            # items = [np.frombuffer(x, dtype=np.float32).reshape(data['shape']) for x in items]
            # cache_name = 'co'
        else:
            items = [np.expand_dims(x, axis=0) for x in data['data']]
        pickle.dump(data['lookup_table'], open(look_table_file_path, 'wb'), protocol=4)
        items = np.concatenate(items, axis=0)
        with h5py.File(hdf5_file_path, 'w') as hf:
            hf.create_dataset(cache_name, data=items)


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
