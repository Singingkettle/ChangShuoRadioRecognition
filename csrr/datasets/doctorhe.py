import copy
import os.path as osp

import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset

from .builder import DATASETS
from ..common import DataContainer as DC
from ..common.utils.path import glob
from ..performance import ClassificationMetricsForSingle


data_format = dict(CD='cos', SQCD='cos', frequency_domain='iqs', time_domain='iqs')


@DATASETS.register_module()
class DoctorHeDataset(Dataset):
    CLASSES = ['BPSK', 'QPSK', '8PSK', '16PSK', '8QAM', '16QAM']

    def __init__(self, data_root, case, data_info, test_mode=False, format=None):
        self.data_root = data_root
        self.case = case
        self.data_info = data_info
        self.test_mode = test_mode
        self.look_table = []
        self.look_table_range = dict()
        self.data_mode = data_format[data_info]
        self.x, self.y = self._load_input()
        self.data_num = None

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

    def _load_input(self):
        label = loadmat(osp.join(self.data_root, self.case, 'label.mat'))
        label = np.squeeze(label['label']) - 1

        data = dict()
        if osp.isdir(osp.join(self.data_root, self.case, self.data_info)):
            data_files = glob(osp.join(self.data_root, self.case, self.data_info), '.mat')
            for data_file in data_files:
                data_key = osp.basename(data_file).split('.')[0]
                data_key = data_key.split('_')
                data_key = data_key[[idx for idx, s in enumerate(data_key) if 'dB' in s][0]]
                data[data_key] = loadmat(data_file)
        else:
            data_key = self.data_info.split('.')[0]
            data_key = data_key.split('_')
            data_key = data_key[[idx for idx, s in enumerate(data_key) if 'dB' in s][0]]
            data[data_key] = loadmat(osp.join(self.data_root, self.case, self.data_info))

        keys = []
        for data_key in data:
            for key in data[data_key]:
                # if 'train' in key or 'valid' in key:
                #     keys.append((data_key, key))
                if self.test_mode:
                    if 'test' in key:
                        keys.append((data_key, key))
                else:
                    if 'train' in key or 'valid' in key:
                        keys.append((data_key, key))

        x = dict()
        y = dict()
        for key in keys:
            if 'train' in key[1]:
                tmp = 'train'
            elif 'valid' in key[1]:
                tmp = 'valid'
            elif '2tp' in key[1]:
                tmp = '2tp'
            else:
                tmp = '5tp'
            x[f'{tmp}/{key[0]}'] = np.squeeze(data[key[0]][key[1]])
            y[f'{tmp}/{key[0]}'] = copy.deepcopy(label)

            start = len(self.look_table)
            for i in range(x[f'{tmp}/{key[0]}'].size):
                item = dict(index=i, key=f'{tmp}/{key[0]}')
                self.look_table.append(item)
            end = start + x[f'{tmp}/{key[0]}'].size
            self.look_table_range[f'{tmp}/{key[0]}'] = [start, end]

        return x, y

    def _set_group_flag(self):
        """Set flag according to the mean value (ms) of signal data snrs.

        set as group i, where self.SNRS[i]<=ms<self.SNRS[i+1].
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def __len__(self):
        """Total number of samples of data."""

        return len(self.look_table)

    def prepare_train_data(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        item_info = self.look_table[idx]
        results = dict(inputs=dict(), targets=dict(), input_metas=dict())

        x = self.x[item_info['key']][item_info['index']]
        if self.data_mode == 'iqs':
            x = x.T
        results['inputs'][self.data_mode] = np.expand_dims(x, axis=0).astype(
            np.float32)
        results['targets']['modulations'] = np.array(self.y[item_info['key']][item_info['index']], dtype=np.int64)
        results['input_metas'] = DC(dict(file_name=idx), cpu_only=True)
        return results

    def prepare_test_data(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """
        item_info = self.look_table[idx]
        results = dict(inputs=dict(), input_metas=dict())
        x = self.x[item_info['key']][item_info['index']]
        if self.data_mode == 'iqs':
            x = x.T
        results['inputs'][self.data_mode] = np.expand_dims(x, axis=0).astype(
            np.float32)
        results['input_metas'] = DC(dict(file_name=idx), cpu_only=True)
        return results

    def __getitem__(self, idx):

        if self.test_mode:
            data = self.prepare_test_data(idx)
        else:
            data = self.prepare_train_data(idx)

        return data

    def evaluate(self, results, logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            eval_results (dict):
        """

        results = np.stack(results, axis=0)

        eval_results = dict()
        for key in self.look_table_range:
            performance_generator = ClassificationMetricsForSingle(
                results[self.look_table_range[key][0]:self.look_table_range[key][1]], self.y[key], self.CLASSES)
            eval_results[f'{key}_ACC'] = performance_generator.ACC
            F1s = performance_generator.F1
            for mod in F1s:
                eval_results[f'{key}_F1_{mod}'] = F1s[mod]


        acc_2tp = 0.0
        f1_2tp = 0.0
        num_2tp = 0.0
        acc_5tp = 0.0
        f1_5tp = 0.0
        num_5tp = 0.0
        for key in eval_results:
            if '2tp' in key:
                if 'ACC' in key:
                    acc_2tp += eval_results[key]
                else:
                    f1_2tp += eval_results[key]
                num_2tp += 1
            else:
                if 'ACC' in key:
                    acc_5tp += eval_results[key]
                else:
                    f1_5tp += eval_results[key]
                num_5tp += 1

        if num_2tp > 0.0:
            eval_results['ACC_2tp'] = acc_2tp / num_2tp
            eval_results['F1_2tp'] = f1_2tp / num_2tp
        if num_5tp > 0.0:
            eval_results['ACC_5tp'] = acc_5tp / num_5tp
            eval_results['F1_5tp'] = f1_5tp / num_2tp
        return eval_results
