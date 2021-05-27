import json
import os.path as osp

import numpy as np
import scipy.io


class ISSDataBase(object):
    def __init__(self, root_dir, version, snr, mode, data_ratios, data_name='data.mat'):

        self.name = 'WTISS'
        self.root_dir = root_dir
        self.version = version
        self.snr = snr
        self.mode = mode
        self.data_ratios = data_ratios
        self.data_name = data_name
        self.data_dir = osp.join(
            self.root_dir, self.name, self.version, self.snr, self.mode)
        self.train_num = 0
        self.val_num = 0
        self.test_num = 0

    def save_as_json(self, data, data_path):
        json.dump(data, open(data_path, 'w'), indent=4, sort_keys=True)

    def process_data(self):
        try:
            print('Start converting data {}/{}/{}/{}'.format(self.name,
                                                             self.version, self.snr, self.mode))
            all_data = scipy.io.loadmat(
                osp.join(self.data_dir, self.data_name))
            all_data = all_data['signal']

            data_num = all_data.shape[0]
            self.train_num = int(data_num * self.data_ratios[0])
            self.val_num = int(data_num * self.data_ratios[1])
            self.test_num = data_num - self.train_num - self.val_num

            np.savetxt(osp.join(self.data_dir, 'train.txt'),
                       all_data[:self.train_num, :])
            np.savetxt(osp.join(self.data_dir, 'val.txt'),
                       all_data[self.train_num:self.train_num + self.val_num, :])
            np.savetxt(osp.join(self.data_dir, 'test.txt'),
                       all_data[self.train_num + self.val_num:, :])

            train_data = dict(data_name='train.txt', data_num=self.train_num)
            val_data = dict(data_name='val.txt', data_num=self.val_num)
            test_data = dict(data_name='test.txt', data_num=self.test_num)

            print(
                'Save train, val, test annotation json for the data set {}/{}/{}/{}'.format(self.name, self.version,
                                                                                            self.snr, self.mode))
            self.save_as_json(train_data, osp.join(
                self.data_dir, 'train.json'))
            self.save_as_json(val_data, osp.join(self.data_dir, 'val.json'))
            self.save_as_json(test_data, osp.join(self.data_dir, 'test.json'))

        except Exception as e:
            print('Error Message is: {}'.format(e))
            raise RuntimeError(
                'Converting data {}/{}/{}/{} failed'.format(self.name, self.version, self.snr, self.mode))
