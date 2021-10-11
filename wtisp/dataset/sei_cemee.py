#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Project: wtisignalprocessing
File: sei_cemee.py
Author: Citybuster
Time: 2021/8/21 10:06
Email: chagshuo@bupt.edu.cn
"""

import os

import h5py
import numpy as np
from torch.utils.data import Dataset

from .builder import DATASETS


@DATASETS.register_module()
class SEICEMEEDataset(Dataset):
    CLASSES = None

    def __init__(self, data_root, mat_file, test_mode=False):
        self.test_mode = test_mode

        # load annotations
        self.X, self.Y = self.load_mat(os.path.join(data_root, mat_file))
        self.index_class_dict = {index: label for index, label in enumerate(self.Y)}

        self.CLASSES = list(set(self.Y))
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

    def _set_group_flag(self):
        """Set flag according to the mean value (ms) of signal data snrs.

        set as group i, where self.SNRS[i]<=ms<self.SNRS[i+1].
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def _generate_mod_labels(self):
        self.mix_mod_labels = np.zeros(
            (len(self), len(self.CLASSES)), dtype=np.float32)
        for idx in range(len(self)):
            ann = self.get_ann_info(idx)
            labels = ann['mod_labels']
            for class_index in labels:
                self.mix_mod_labels[idx, class_index] = 1

    def __len__(self):
        """Total number of samples of data."""
        return len(self.Y)

    def load_mat(self, mat_file):
        """Load annotation from annotation file."""
        with h5py.File(mat_file, 'r') as f:
            X = np.array(f['X'], dtype=np.float32)
            X = np.transpose(X, (2, 1, 0))
            Y = np.array(f['Y'], dtype=np.int)
            Y = list(Y[0, :])

        return X, Y

    def __getitem__(self, idx):

        if self.test_mode:
            data = self.prepare_test_data(idx)
        else:
            data = self.prepare_train_data(idx)

        return data

    def get_cat_ids(self, idx):
        """Get category ids by idx.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the data of specified idx.
        """

        return self.Y[idx]

    def prepare_train_data(self, idx):
        """Get training data.

        Args.
         idx (int): Index of data

        Returns:
            dict: Training data and annotation
        """
        x = self.X[idx, :, :]
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)
        x = np.transpose(x, (2, 0, 1))

        y = self.Y[idx]
        y = np.array([y], dtype=np.int64)

        return dict(iqs=x, mod_labels=y)

    def prepare_test_data(self, idx):
        """Get testing data.

        Args.
         idx (int): Index of data

        Returns:
            dict: Testing data and annotation
        """
        x = self.X[idx, :, :]
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)
        x = np.transpose(x, (2, 0, 1))

        return dict(iqs=x)

    def _evaluate_class(self, results, prefix=''):
        conf = np.zeros((len(self.CLASSES), len(self.CLASSES)), dtype=np.float64)

        for idx, label in enumerate(self.Y):
            predict_class_index = int(np.argmax(results[idx, :]))
            conf[label, predict_class_index] += 1

        conf = conf / np.expand_dims(np.sum(conf, axis=1), axis=1)

        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor

        maa = 1.0 * cor / (cor + ncor)
        return {prefix + 'maa': maa}

    def _reshape_result(self, results, num_cols):
        results = [result.reshape(1, num_cols) for result in results]
        results = np.concatenate(results, axis=0)
        results = np.reshape(results, (len(self), -1))
        return results

    def process_single_head(self, results, prefix=''):

        results = self._reshape_result(results, len(self.CLASSES))
        eval_results = self._evaluate_class(results, prefix=prefix)

        return eval_results

    def evaluate(self, results, logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        """
        eval_results = self.process_single_head(results, prefix='common/')

        return eval_results

    def format_out(self, out_dir, results):
        """Format the results to json and save.

        Args:
            out_dir (str): the out dir to save the json file
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
                format(len(results), len(self)))

        save_path = os.path.join(out_dir, 'pre.npy')
        results = [result.reshape(1, len(self.CLASSES)) for result in results]
        results = np.concatenate(results, axis=0)
        results = np.reshape(results, (len(self), -1))

        save_results = np.zeros((len(self), 1), dtype=np.int)
        for i in range(len(self)):
            save_results[i, :] = int(np.argmax(results[i, :]))

        np.save(save_path, save_results)
