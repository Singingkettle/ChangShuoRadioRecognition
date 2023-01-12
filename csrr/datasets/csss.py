import os
from abc import ABCMeta, abstractmethod

import numpy as np
import scipy.io as sio
from scipy.special import expit, softmax
from torch.utils.data import Dataset

from .builder import DATASETS
from .merge.methods import get_merge_weight_by_grid_search
from .utils import format_results, reshape_results


@DATASETS.register_module()
class CSSSMat(Dataset, metaclass=ABCMeta):
    CLASSES = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM"]
    SNRS = list(range(-8, 22, 2))

    def __init__(self, file_info, data_root=None, test_mode=False):
        file_prefixes = file_info['file_prefixes']
        snrs = file_info['snrs']

        self.iqs = []
        self.iqs_ = []
        self.mod_labels = []
        self.band_labels = []
        self.snrs = snrs
        self.num_snr = len(snrs)
        for file_prefix in file_prefixes:
            for snr in snrs:
                file_path = os.path.join(data_root, f'{file_prefix}_{snr:02d}.mat')
                data = sio.loadmat(file_path)
                iqs = data[file_prefix][0, 0][0].astype(np.float32)
                iqs_ = data[file_prefix][0, 0][1].astype(np.float32)
                mod_labels = data[file_prefix][0, 0][2].astype(np.int)
                band_labels = data[file_prefix][0, 0][3].astype(np.float32)
                self.iqs.append(iqs)
                self.iqs_.append(iqs_)
                self.mod_labels.append(mod_labels)
                self.band_labels.append(band_labels)

        self.iqs = np.concatenate(self.iqs, axis=0)
        self.iqs_ = np.concatenate(self.iqs_, axis=0)
        self.mod_labels = np.concatenate(self.mod_labels, axis=0)
        self.band_labels = np.concatenate(self.band_labels, axis=0)
        self.num = self.iqs.shape[0]

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

    def __len__(self):
        return self.num

    def _set_group_flag(self):
        """Set flag according to the mean value (ms) of signal data snrs.

        set as group i, where self.SNRS[i]<=ms<self.SNRS[i+1].
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    @abstractmethod
    def prepare_train_data(self, idx):
        pass

    @abstractmethod
    def prepare_test_data(self, idx):
        pass

    def __getitem__(self, idx):

        if self.test_mode:
            data = self.prepare_test_data(idx)
        else:
            data = self.prepare_train_data(idx)

        return data

    @abstractmethod
    def evaluate(self, results, logger=None):
        pass


@DATASETS.register_module()
class CSSSBCE(CSSSMat):
    def __int__(self, file_info, data_root=None, test_mode=False):
        super(CSSSBCE).__int__(file_info, data_root, test_mode)

        mod_labels = np.zeros((len(self), 5 * 9), dtype=np.float32)
        for idx in range(len(self)):
            ms = self.mod_labels[idx, :]
            bs = self.band_labels[idx, :]
            for i in range(2):
                col_index = bs[i] * 5 + ms[i]
                mod_labels[idx, col_index] = 1.0

        self.mod_labels = mod_labels

    def prepare_train_data(self, idx):
        x = self.iqs[idx, :, :]
        y = self.mod_labels[idx, :]

        return dict(iqs=x, mod_labels=y)

    def prepare_test_data(self, idx):
        x = self.iqs[idx, :, :]

        return dict(iqs=x)

    def evaluate(self, results, logger=None):
        results = format_results(results)
        pre_matrix = []
        for key_str in results:
            pre_data = results[key_str]
            pre_data = reshape_results(pre_data, 9 * 5)
            pre_data = expit(pre_data)
            pre_data = pre_data[None, :, :]
            pre_matrix.append(pre_data)
        pre_matrix = np.concatenate(pre_matrix, axis=0)
        search_weight_list = get_merge_weight_by_grid_search(len(results), self.grid_step)
        eval_results = dict()

        def _eval(res):
            y_ = np.zeros((1, 9 * 5), dtype=np.float32)
            res = np.reshape(res, [9, 5])
            max_scores = np.max(res, axis=1)
            max_scores_index = np.argmax(res, axis=1)
            sorted_index = np.argsort(max_scores)
            for index in sorted_index[-2:]:
                y_[0, index * 5 + max_scores_index[index]] = 1.0

            return y_

        cur_max_accuracy = 0
        cur_search_weight = None
        cur_dy = None
        for search_weight in search_weight_list:
            search_weight = np.array(search_weight)
            search_weight = np.reshape(search_weight, (1, -1))
            tmp_merge_matrix = np.dot(search_weight, np.reshape(pre_matrix, (len(results), -1)))
            tmp_merge_matrix = np.reshape(tmp_merge_matrix, (-1, 9 * 5))
            tmp_merge_matrix = tmp_merge_matrix.tolist()
            y_ = list(map(_eval, tmp_merge_matrix))
            y_ = np.concatenate(y_, axis=0)
            dy = self.mod_labels - y_
            dy = np.sum(np.abs(dy), axis=1)
            accuracy = (len(self) - np.count_nonzero(dy)) * 1.0 / len(self)
            if accuracy >= cur_max_accuracy:
                cur_max_accuracy = accuracy
                cur_search_weight = search_weight
                cur_dy = dy

        print('The best search weight is:')
        print(cur_search_weight)
        print('\n')
        cur_dy = np.reshape(cur_dy, [self.num_snr, -1])
        for snr_index in range(self.num_snr):
            accuracy = (len(self) / self.num_snr - np.count_nonzero(cur_dy[snr_index, :])) * 1.0 / len(
                self) / self.num_snr
            eval_results[f'snr_{self.snrs[snr_index]:02d}dB'] = accuracy
        eval_results['snr_mean_all'] = cur_max_accuracy

        return eval_results


@DATASETS.register_module()
class CSSSDetTwoStage(CSSSMat):
    def __int__(self, file_info, num_anchor=4, data_root=None, test_mode=False):
        super(CSSSDetTwoStage).__int__(file_info, data_root, test_mode)

        self.num_anchor = num_anchor
        fft_iqs = np.zeros((len(self), 2, 4096), dtype=np.float32)
        mod_labels = np.zeros((len(self), 9), dtype=np.float32) + 5
        band_labels = np.zeros((len(self), 9), dtype=np.float32)
        self.gt = np.zeros((len(self), 9 * 5), dtype=np.float32)
        for idx in range(len(self)):
            iq = self.iqs[idx, 0, :] + 1j * self.iqs[idx, 1, :]
            iq = np.fft.fft(iq)
            iq = np.fft.fftshift(iq)
            iq = np.vstack((np.real(iq), np.imag(iq)))
            fft_iqs[idx, :, :] = iq
            band_labels[idx, self.band_labels[idx, 0]] = 1
            band_labels[idx, self.band_labels[idx, 1]] = 1
            mod_labels[idx, 0] = self.mod_labels[idx, 0]
            mod_labels[idx, 1] = self.mod_labels[idx, 1]
            for i in range(2):
                col_index = band_labels[i] * 5 + mod_labels[i]
                self.gt[idx, col_index] = 1.0

        self.iqs = fft_iqs
        self.band_labels = band_labels
        self.mod_labels = mod_labels

    def prepare_train_data(self, idx):
        x = self.iqs[idx, :, :]
        x_ = self.iqs_[idx, :, :, :]
        my = self.mod_labels[idx, :]
        by = self.band_labels[idx, :]

        return dict(iqs=x, iqs_=x_, mod_labels=my, band_labels=by)

    def evaluate(self, results, logger=None):
        results = format_results(results)
        eval_results = dict()

        def _eval(res_band, res_mod):
            y_ = np.zeros((1, 9 * 5), dtype=np.float32)
            anchor_scores = []
            for anchor_index in range(self.num_anchor):
                anchor_scores.append(res_band[anchor_index] * max(res_mod[anchor_index]))

            anchor_scores = np.array(anchor_scores)
            sorted_anchor_index = np.argsort(anchor_scores)
            for anchor_index in sorted_anchor_index[-2:]:
                y_[0, res_band[anchor_index] * 5 + np.argmax(res_mod[anchor_index])] = 1.0

            return y_

        y_ = list(map(_eval, results['Band'], results['Mod']))
        dy = self.gt - y_
        dy = np.sum(np.abs(dy), axis=1)
        for snr_index in range(self.num_snr):
            accuracy = (len(self) / self.num_snr - np.count_nonzero(dy[snr_index, :])) * 1.0 / len(
                self) / self.num_snr
            eval_results[f'snr_{self.snrs[snr_index]:02d}dB'] = accuracy
        eval_results['snr_mean_all'] = (len(self) - np.count_nonzero(dy)) * 1.0 / len(self)

        return eval_results


@DATASETS.register_module()
class CSSSDetSingleStage(CSSSMat):
    def __int__(self, file_info, num_anchor=4, data_root=None, test_mode=False):
        super(CSSSDetSingleStage).__int__(file_info, data_root, test_mode)
        self.num_anchor = num_anchor
        mod_labels = np.zeros((len(self), 9), dtype=np.int64) + -1
        band_labels = np.zeros((len(self), 9), dtype=np.float32)
        self.gt = np.zeros((len(self), 9 * 5), dtype=np.float32)
        for idx in range(len(self)):
            band_labels[idx, self.band_labels[idx, 0]] = 1
            band_labels[idx, self.band_labels[idx, 1]] = 1
            mod_labels[idx, self.band_labels[idx, 0]] = self.mod_labels[idx, 0]
            mod_labels[idx, self.band_labels[idx, 1]] = self.mod_labels[idx, 1]
            for i in range(2):
                col_index = band_labels[i] * 5 + mod_labels[i]
                self.gt[idx, col_index] = 1.0

        self.band_labels = band_labels
        self.mod_labels = mod_labels

    def prepare_train_data(self, idx):
        x = self.iqs[idx, :, :]
        my = self.mod_labels[idx, :]
        by = self.band_labels[idx, :]

        return dict(iqs=x, mod_labels=my, band_labels=by)

    def evaluate(self, results, logger=None):
        results = format_results(results)
        eval_results = dict()

        def _eval(res_band, res_mod):
            y_ = np.zeros((1, 9 * 5), dtype=np.float32)
            anchor_scores = []
            for anchor_index in range(4):
                anchor_scores.append(res_band[anchor_index] * max(res_mod[anchor_index]))

            anchor_scores = np.array(anchor_scores)
            sorted_anchor_index = np.argsort(anchor_scores)
            for anchor_index in sorted_anchor_index[-2:]:
                y_[0, res_band[anchor_index] * 5 + np.argmax(res_mod[anchor_index])] = 1.0

            return y_

        y_ = list(map(_eval, results['Band'], results['Mod']))
        dy = self.gt - y_
        dy = np.sum(np.abs(dy), axis=1)
        for snr_index in range(self.num_snr):
            accuracy = (len(self) / self.num_snr - np.count_nonzero(dy[snr_index, :])) * 1.0 / len(
                self) / self.num_snr
            eval_results[f'snr_{self.snrs[snr_index]:02d}dB'] = accuracy
        eval_results['snr_mean_all'] = (len(self) - np.count_nonzero(dy)) * 1.0 / len(self)

        return eval_results
