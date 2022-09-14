import os.path as osp

import h5py
import numpy as np
from scipy.special import expit
from torch.utils.data import Dataset

from .builder import DATASETS
from .merge.methods import get_merge_weight_by_grid_search
from .utils import format_results, reshape_results


@DATASETS.register_module()
class GBSenseBasic(Dataset):
    """GBSense 2022 dataset for modulation classification. http://www.gbsense.net/challenge/
    Args:
        file_name (str): HDF5 file name for IQ and Label.
        data_root (str, optional): Data root for HDF5 file.
        test_mode (bool, optional): If set True, annotation will not be loaded.
    """
    CLASSES = ["APSK16", "APSK32", "APSK64", "ASK8", "BPSK", "OQPSK",
               "PSK16", "PSK8", "QAM128", "QAM16", "QAM256", "QAM64", "QPSK"]

    def __init__(self, file_name, grid_step=0.01, data_root=None, test_mode=False):

        self.grid_step = grid_step
        # load data
        if isinstance(file_name, list):
            x = []
            y = []
            for h5_file in file_name:
                data = h5py.File(osp.join(data_root, h5_file))
                x.append(data['X'][:, :, :])
                y.append(data['Y'][:, :])
            self.X = np.concatenate(x, axis=0)
            self.Y = np.concatenate(y, axis=1)
        else:
            data = h5py.File(osp.join(data_root, file_name))
            self.X = data['X'][:, :, :]
            self.Y = data['Y'][:, :]

        self.test_mode = test_mode
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        del data

    def __len__(self):
        return self.X.shape[0]

    def _set_group_flag(self):
        """Set flag according to the mean value (ms) of signal data snrs.

        set as group i, where self.SNRS[i]<=ms<self.SNRS[i+1].
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def prepare_test_data(self, idx):
        x = self.X[idx, :, :]
        x = (x - np.mean(x, axis=0).reshape(1, 16)) / np.std(x, axis=0).reshape(1, 16)
        iqs = np.transpose(x)
        iqs = np.expand_dims(iqs, axis=1)
        iqs = iqs.astype(np.float32)
        return dict(iqs=iqs)

    def prepare_train_data(self, idx):
        x = self.X[idx, :, :]
        x = (x - np.mean(x, axis=0).reshape(1, 16)) / np.std(x, axis=0).reshape(1, 16)
        y = self.Y[idx, :] - 1
        iqs = np.transpose(x)
        iqs = np.expand_dims(iqs, axis=1)
        iqs = iqs.astype(np.float32)

        mod_labels = y.astype(np.int64)
        return dict(iqs=iqs, mod_labels=mod_labels)

    def __getitem__(self, idx):

        if self.test_mode:
            data = self.prepare_test_data(idx)
        else:
            data = self.prepare_train_data(idx)

        return data

    def get_confusion_matrix(self, results):
        confusion_matrix = np.zeros((len(self.CLASSES), len(self.CLASSES)), dtype=np.float64)
        for idx in range(results.shape[0]):
            predict_label = int(np.argmax(results[idx, :]))
            gt_label = int(self.Y[idx, :] - 1)
            confusion_matrix[gt_label, predict_label] += 1
        return confusion_matrix

    def evaluate(self, results, logger=None):
        results = format_results(results)
        pre_matrix = []
        for key_str in results:
            pre_data = results[key_str]
            pre_data = reshape_results(pre_data, len(self.CLASSES))
            pre_data = expit(pre_data)
            pre_data = pre_data[None, :, :]
            pre_matrix.append(pre_data)

        pre_matrix = np.concatenate(pre_matrix, axis=0)
        search_weight_list = get_merge_weight_by_grid_search(len(results), self.grid_step)
        cur_max_accuracy = 0
        cur_search_weight = None
        eval_results = dict()
        for search_weight in search_weight_list:
            search_weight = np.array(search_weight)
            search_weight = np.reshape(search_weight, (1, -1))
            tmp_merge_matrix = np.dot(search_weight, np.reshape(pre_matrix, (len(results), -1)))
            tmp_merge_matrix = np.reshape(tmp_merge_matrix, (-1, len(self.CLASSES)))
            conf = self.get_confusion_matrix(tmp_merge_matrix)
            cor = np.sum(np.diag(conf))
            ncor = np.sum(conf) - cor
            tmp_accuracy = 1.0 * cor / (cor + ncor)
            if cur_max_accuracy < tmp_accuracy:
                cur_max_accuracy = tmp_accuracy
                cur_search_weight = search_weight

        print('The best search weight is:')
        print(cur_search_weight)
        print('\n')
        eval_results['mean_all'] = cur_max_accuracy
        return eval_results


@DATASETS.register_module()
class GBSenseAdvanced(GBSenseBasic):

    def prepare_train_data(self, idx):
        x = self.X[idx, :, :]
        x = (x - np.mean(x, axis=0).reshape(1, 16)) / np.std(x, axis=0).reshape(1, 16)
        y = self.Y[idx, :]
        mod_labels = np.zeros(24 * 13, dtype=np.float32)
        for i, val in enumerate(y):
            if val > 0:
                mod_labels[i * 13 + val - 1] = 1.0

        iqs = np.transpose(x)
        iqs = np.expand_dims(iqs, axis=1)
        iqs = iqs.astype(np.float32)

        return dict(iqs=iqs, mod_labels=mod_labels)

    def __getitem__(self, idx):

        if self.test_mode:
            data = self.prepare_test_data(idx)
        else:
            data = self.prepare_train_data(idx)

        return data

    def get_confusion_matrix(self, results):
        confusion_matrix = np.zeros((24, len(self.CLASSES), len(self.CLASSES)), dtype=np.float64)
        for idx in range(results.shape[0]):
            predict_label = results[idx, :]
            max_indices = np.argpartition(predict_label, -2)[-2:]
            gt_label = self.Y[idx, :]
            for index in max_indices:
                c = index // 13
                m = index % 13
                confusion_matrix[c, gt_label[c] - 1, m] += 1

        return confusion_matrix

    def evaluate(self, results, logger=None):
        results = format_results(results)
        pre_matrix = []
        for key_str in results:
            pre_data = results[key_str]
            pre_data = reshape_results(pre_data, len(self.CLASSES))
            pre_data = expit(pre_data)
            pre_data = pre_data[None, :, :]
            pre_matrix.append(pre_data)

        pre_matrix = np.concatenate(pre_matrix, axis=0)
        search_weight_list = get_merge_weight_by_grid_search(len(results), self.grid_step)
        cur_max_accuracy = 0
        cur_search_weight = None
        eval_results = dict()
        for search_weight in search_weight_list:
            search_weight = np.array(search_weight)
            search_weight = np.reshape(search_weight, (1, -1))
            tmp_merge_matrix = np.dot(search_weight, np.reshape(pre_matrix, (len(results), -1)))
            tmp_merge_matrix = np.reshape(tmp_merge_matrix, (-1, len(self.CLASSES)))
            confusion_matrix = self.get_confusion_matrix(tmp_merge_matrix)
            conf = np.sum(confusion_matrix, axis=0) / 24
            cor = np.sum(np.diag(conf))
            ncor = np.sum(conf) - cor
            tmp_accuracy = 1.0 * cor / (cor + ncor)
            if cur_max_accuracy < tmp_accuracy:
                cur_max_accuracy = tmp_accuracy
                cur_search_weight = search_weight
                for channel_index in range(24):
                    conf = confusion_matrix[channel_index, :, :]
                    cor = np.sum(np.diag(conf))
                    ncor = np.sum(conf) - cor
                    eval_results['channel_{:d}'.format(channel_index + 1)] = 1.0 * cor / (
                            cor + ncor + np.finfo(np.float64).eps)

                conf = np.sum(confusion_matrix, axis=0) / 24
                cor = np.sum(np.diag(conf))
                ncor = np.sum(conf) - cor
                eval_results['channel_mean_all'] = 1.0 * cor / (cor + ncor + np.finfo(np.float64).eps)

            print('The best search weight is:')
            print(cur_search_weight)
            print('\n')

        return eval_results


@DATASETS.register_module()
class GBSenseAdvanced2(GBSenseBasic):

    def prepare_train_data(self, idx):
        x = self.X[idx, :, :]
        x = (x - np.mean(x, axis=0).reshape(1, 16)) / np.std(x, axis=0).reshape(1, 16)
        y = self.Y[idx, :]
        mod_labels = np.zeros(13, dtype=np.float32)
        channel_labels = np.zeros(24, dtype=np.float32)
        order_labels = np.zeros(1, dtype=np.float32)
        l = []
        for i, val in enumerate(y):
            if val > 0:
                mod_labels[val - 1] = 1
                channel_labels[i] = 1
                l.append(val)

        iqs = np.transpose(x)
        iqs = np.expand_dims(iqs, axis=1)
        iqs = iqs.astype(np.float32)

        return dict(iqs=iqs, mod_labels=mod_labels)

    def __getitem__(self, idx):

        if self.test_mode:
            data = self.prepare_test_data(idx)
        else:
            data = self.prepare_train_data(idx)

        return data

    def get_confusion_matrix(self, results):
        confusion_matrix = np.zeros((24, len(self.CLASSES), len(self.CLASSES)), dtype=np.float64)
        for idx in range(results.shape[0]):
            predict_label = results[idx, :]
            max_indices = np.argpartition(predict_label, -2)[-2:]
            gt_label = self.Y[idx, :]
            for index in max_indices:
                c = index // 13
                m = index % 13
                confusion_matrix[c, gt_label[c] - 1, m] += 1

        return confusion_matrix

    def evaluate(self, results, logger=None):
        results = format_results(results)
        pre_matrix = []
        for key_str in results:
            pre_data = results[key_str]
            pre_data = reshape_results(pre_data, len(self.CLASSES))
            pre_data = expit(pre_data)
            pre_data = pre_data[None, :, :]
            pre_matrix.append(pre_data)

        pre_matrix = np.concatenate(pre_matrix, axis=0)
        search_weight_list = get_merge_weight_by_grid_search(len(results), self.grid_step)
        cur_max_accuracy = 0
        cur_search_weight = None
        eval_results = dict()
        for search_weight in search_weight_list:
            search_weight = np.array(search_weight)
            search_weight = np.reshape(search_weight, (1, -1))
            tmp_merge_matrix = np.dot(search_weight, np.reshape(pre_matrix, (len(results), -1)))
            tmp_merge_matrix = np.reshape(tmp_merge_matrix, (-1, len(self.CLASSES)))
            confusion_matrix = self.get_confusion_matrix(tmp_merge_matrix)
            conf = np.sum(confusion_matrix, axis=0) / 24
            cor = np.sum(np.diag(conf))
            ncor = np.sum(conf) - cor
            tmp_accuracy = 1.0 * cor / (cor + ncor)
            if cur_max_accuracy < tmp_accuracy:
                cur_max_accuracy = tmp_accuracy
                cur_search_weight = search_weight
                for channel_index in range(24):
                    conf = confusion_matrix[channel_index, :, :]
                    cor = np.sum(np.diag(conf))
                    ncor = np.sum(conf) - cor
                    eval_results['channel_{:d}'.format(channel_index + 1)] = 1.0 * cor / (
                            cor + ncor + np.finfo(np.float64).eps)

                conf = np.sum(confusion_matrix, axis=0) / 24
                cor = np.sum(np.diag(conf))
                ncor = np.sum(conf) - cor
                eval_results['channel_mean_all'] = 1.0 * cor / (cor + ncor + np.finfo(np.float64).eps)

            print('The best search weight is:')
            print(cur_search_weight)
            print('\n')

        return eval_results
