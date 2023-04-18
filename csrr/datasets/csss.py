import copy
import os
from abc import ABCMeta, abstractmethod

import numpy as np
import scipy.io as sio
from scipy.special import expit
from torch.utils.data import Dataset

from .builder import DATASETS
from .merge.methods import get_merge_weight_by_grid_search
from ..common import DataContainer as DC


class BaseCSSS(Dataset, metaclass=ABCMeta):
    CLASSES = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM"]

    def __init__(self, file_info, data_root, test_mode):
        file_prefixes = file_info['file_prefixes']
        snrs = file_info['snrs']

        self.iqs = []
        self.iqs_ = []
        self.mod_labels = []
        self.band_labels = []
        self.snrs = snrs
        self.num_snr = len(snrs)
        self.test_mode = test_mode
        self.snr_range = [0]
        self.num_mod = len(self.CLASSES)

        for snr in snrs:
            new_index = 0
            for file_prefix in file_prefixes:
                file_path = os.path.join(data_root, f'{file_prefix}_{snr:02d}.mat')
                data = sio.loadmat(file_path)
                iqs = data[file_prefix]['x'][0, 0].astype(np.float32)
                iqs_ = data[file_prefix]['x_'][0, 0].astype(np.float32)
                mod_labels = data[file_prefix]['m'][0, 0].astype(np.int64)
                band_labels = data[file_prefix]['b'][0, 0].astype(np.int64)
                new_index = iqs.shape[0] + new_index
                self.iqs.append(iqs)
                self.iqs_.append(iqs_)
                self.mod_labels.append(mod_labels)
                self.band_labels.append(band_labels)
            new_index = new_index + self.snr_range[-1]
            self.snr_range.append(new_index)

        self.iqs = np.concatenate(self.iqs, axis=0)
        self.iqs_ = np.concatenate(self.iqs_, axis=0)
        self.mod_labels = np.concatenate(self.mod_labels, axis=0)
        self.band_labels = np.concatenate(self.band_labels, axis=0)
        self.num = self.iqs.shape[0]
        self.num_band = self.iqs_.shape[1]
        self.time_length = self.iqs.shape[-1]

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
class CSSSBCE(BaseCSSS):
    def __int__(self, file_info, data_root=None, test_mode=False):
        super(CSSSBCE, self).__init__(file_info, data_root, test_mode)

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
class CSSSDetTwoStage(BaseCSSS):
    def __init__(self, file_info, data_root=None, test_mode=False):
        super(CSSSDetTwoStage, self).__init__(file_info, data_root, test_mode)

        fft_iqs = np.zeros((len(self), 2, self.time_length), dtype=np.float32)
        band_labels = np.zeros((len(self), self.num_band), dtype=np.float32)
        self.gt = np.zeros((len(self), self.num_band * self.num_mod), dtype=np.float32)
        self.occupied_bands = copy.deepcopy(self.band_labels)
        for idx in range(len(self)):
            iq = self.iqs[idx, 0, :] + 1j * self.iqs[idx, 1, :]
            iq = np.fft.fft(iq)
            iq = np.fft.fftshift(iq)
            iq = np.vstack((np.real(iq), np.imag(iq)))
            fft_iqs[idx, :, :] = iq
            band_labels[idx, self.band_labels[idx, 0]] = 1
            band_labels[idx, self.band_labels[idx, 1]] = 1
            for i in range(2):
                col_index = self.band_labels[idx, i] * self.num_mod + self.mod_labels[idx, i]
                self.gt[idx, col_index] = 1.0

        self.iqs = fft_iqs
        self.band_labels = band_labels

    def prepare_test_data(self, idx):
        x = self.iqs[idx, :, :]
        x_ = self.iqs_[idx, :, :, :]

        inputs = dict(iqs=x, iqs_=x_)
        input_metas = DC(dict(file_name=0), cpu_only=True)

        return dict(inputs=inputs, input_metas=input_metas)

    def prepare_train_data(self, idx):
        x = self.iqs[idx, :, :]
        x_ = self.iqs_[idx, :, :, :]
        x_ = [x_[self.occupied_bands[idx, 0], :, :], x_[self.occupied_bands[idx, 1], :, :]]
        x_ = np.concatenate(x_, axis=0)

        my = self.mod_labels[idx, :]
        by = self.band_labels[idx, :]

        inputs = dict(iqs=x, iqs_=x_)
        targets = dict(mod_labels=my, band_labels=by)
        input_metas = DC(dict(file_name=0), cpu_only=True)

        return dict(inputs=inputs, input_metas=input_metas, targets=targets)

    def evaluate(self, results, logger=None):
        results = format_results(results)
        eval_results = dict()

        def _eval(res_band, res_mod):
            y_ = np.zeros((1, self.num_mod * self.num_band), dtype=np.float32)
            sorted_anchor_index = np.argsort(res_band)
            for anchor_index in sorted_anchor_index[-2:]:
                y_[0, anchor_index * self.num_mod + np.argmax(res_mod[anchor_index, :])] = 1.0

            return y_

        y_ = list(map(_eval, results['Band'], results['Mod']))
        y_ = np.concatenate(y_, axis=0)
        dy = self.gt - y_
        dy = np.sum(np.abs(dy), axis=1)
        for snr_index in range(self.num_snr):
            snr_left_index = self.snr_range[snr_index]
            snr_right_index = self.snr_range[snr_index + 1]
            accuracy = (snr_right_index - snr_left_index - np.count_nonzero(
                dy[snr_left_index:snr_right_index])) * 1.0 / (snr_right_index - snr_left_index)
            eval_results[f'acc_snr_{self.snrs[snr_index]:02d}dB'] = accuracy
        eval_results['acc_all_snr'] = (len(self) - np.count_nonzero(dy)) * 1.0 / len(self)

        y_ = np.reshape(y_, [-1, self.num_band, self.num_mod])
        y_ = np.sum(y_, axis=2)
        dy = self.band_labels - y_
        dy = np.sum(np.abs(dy), axis=1)
        for snr_index in range(self.num_snr):
            snr_left_index = self.snr_range[snr_index]
            snr_right_index = self.snr_range[snr_index + 1]
            accuracy = (snr_right_index - snr_left_index - np.count_nonzero(
                dy[snr_left_index:snr_right_index])) * 1.0 / (snr_right_index - snr_left_index)
            eval_results[f'band_acc_snr_{self.snrs[snr_index]:02d}dB'] = accuracy
        eval_results['band_acc_all_snr'] = (len(self) - np.count_nonzero(dy)) * 1.0 / len(self)

        return eval_results


@DATASETS.register_module()
class CSSSDetSingleStage(BaseCSSS):
    def __init__(self, file_info, data_root=None, test_mode=False):
        super(CSSSDetSingleStage, self).__init__(file_info, data_root, test_mode)
        fft_iqs = np.zeros((len(self), 2, self.time_length), dtype=np.float32)
        mod_labels = np.ones(len(self) * self.num_band, dtype=np.int64) * self.num_mod
        band_labels = np.zeros((len(self), self.num_band), dtype=np.float32)
        self.gt = np.zeros((len(self), self.num_mod * self.num_band), dtype=np.float32)
        for idx in range(len(self)):
            iq = self.iqs[idx, 0, :] + 1j * self.iqs[idx, 1, :]
            iq = np.fft.fft(iq)
            iq = np.fft.fftshift(iq)
            iq = np.vstack((np.real(iq), np.imag(iq)))
            fft_iqs[idx, :, :] = iq
            band_labels[idx, self.band_labels[idx, 0]] = 1
            band_labels[idx, self.band_labels[idx, 1]] = 1
            mod_labels[idx * self.num_band + self.band_labels[idx, 0]] = self.mod_labels[idx, 0]
            mod_labels[idx * self.num_band + self.band_labels[idx, 1]] = self.mod_labels[idx, 1]
            for i in range(2):
                col_index = self.band_labels[idx, i] * self.num_mod + self.mod_labels[idx, i]
                self.gt[idx, col_index] = 1.0

        self.time_length = self.time_length // self.num_band
        self.mod_labels = mod_labels
        self.band_labels = band_labels
        fft_iqs = np.transpose(fft_iqs, [1, 0, 2])
        fft_iqs = np.reshape(fft_iqs, [2, -1, self.time_length])
        fft_iqs = np.transpose(fft_iqs, [1, 0, 2])
        self.iqs = fft_iqs
        del self.iqs_

    def prepare_train_data(self, idx):
        x = self.iqs[idx * self.num_band:(idx + 1) * self.num_band, :, :]
        my = self.mod_labels[idx * self.num_band:(idx + 1) * self.num_band]

        inputs = dict(iqs=x)
        targets = dict(labels=my)
        input_metas = DC(dict(file_name=0), cpu_only=True)

        return dict(inputs=inputs, input_metas=input_metas, targets=targets)

    def prepare_test_data(self, idx):
        x = self.iqs[idx * self.num_band:(idx + 1) * self.num_band, :, :]
        inputs = dict(iqs=x)
        input_metas = DC(dict(file_name=0), cpu_only=True)

        return dict(inputs=inputs, input_metas=input_metas)

    def evaluate(self, results, logger=None):
        results = format_results(results)
        eval_results = dict()

        def _eval(res_mod):
            y_ = np.zeros((1, self.num_mod * self.num_band), dtype=np.float32)
            sorted_anchor_index = np.argsort(res_mod[:, -1])
            for anchor_index in sorted_anchor_index[:2]:
                y_[0, anchor_index * self.num_mod + np.argmax(res_mod[anchor_index, :-1])] = 1.0
            return y_

        y_ = list(map(_eval, results['Final']))
        y_ = np.concatenate(y_, axis=0)
        dy = self.gt - y_
        dy = np.sum(np.abs(dy), axis=1)
        for snr_index in range(self.num_snr):
            snr_left_index = self.snr_range[snr_index]
            snr_right_index = self.snr_range[snr_index + 1]
            accuracy = (snr_right_index - snr_left_index - np.count_nonzero(
                dy[snr_left_index:snr_right_index])) * 1.0 / (snr_right_index - snr_left_index)
            eval_results[f'acc_snr_{self.snrs[snr_index]:02d}dB'] = accuracy
        eval_results['acc_all_snr'] = (len(self) - np.count_nonzero(dy)) * 1.0 / len(self)

        y_ = np.reshape(y_, [-1, self.num_band, self.num_mod])
        y_ = np.sum(y_, axis=2)
        dy = self.band_labels - y_
        dy = np.sum(np.abs(dy), axis=1)
        for snr_index in range(self.num_snr):
            snr_left_index = self.snr_range[snr_index]
            snr_right_index = self.snr_range[snr_index + 1]
            accuracy = (snr_right_index - snr_left_index - np.count_nonzero(
                dy[snr_left_index:snr_right_index])) * 1.0 / (snr_right_index - snr_left_index)
            eval_results[f'band_acc_snr_{self.snrs[snr_index]:02d}dB'] = accuracy
        eval_results['band_acc_all_snr'] = (len(self) - np.count_nonzero(dy)) * 1.0 / len(self)

        return eval_results


@DATASETS.register_module()
class PureCSSS(Dataset):
    CLASSES = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM"]

    def __init__(self, file_info, data_root, test_mode=False, use_fft=True):
        file_prefixes = file_info['file_prefixes']
        snrs = file_info['snrs']

        self.iqs = []
        self.mod_labels = []
        self.snrs = snrs
        self.num_snr = len(snrs)
        self.test_mode = test_mode
        self.snr_range = [0]
        self.num_mod = len(self.CLASSES)

        for snr in snrs:
            new_index = 0
            for file_prefix in file_prefixes:
                file_path = os.path.join(data_root, f'{file_prefix}_{snr:02d}.mat')
                data = sio.loadmat(file_path)
                iqs = data[file_prefix]['x'][0, 0].astype(np.float32)
                mod_labels = data[file_prefix]['m'][0, 0].astype(np.int64)
                mod_labels = mod_labels.flatten()
                new_index = iqs.shape[0] + new_index
                self.iqs.append(iqs)
                self.mod_labels.append(mod_labels)
            new_index = new_index + self.snr_range[-1]
            self.snr_range.append(new_index)

        self.iqs = np.concatenate(self.iqs, axis=0)
        self.mod_labels = np.concatenate(self.mod_labels, axis=0)
        self.num = self.iqs.shape[0]
        self.time_length = self.iqs.shape[-1]

        self.gt = np.zeros((len(self), self.num_mod), dtype=np.float32)
        for i in range(len(self)):
            self.gt[i, self.mod_labels[i]] = 1.0

        if use_fft:
            fft_iqs = np.zeros((len(self), 2, 1025), dtype=np.float32)
            for idx in range(len(self)):
                iq = self.iqs[idx, 0, :] + 1j * self.iqs[idx, 1, :]
                iq = np.fft.fft(iq)
                iq = np.fft.fftshift(iq)
                iq = np.vstack((np.real(iq), np.imag(iq)))
                fft_iqs[idx, :, :] = iq
            self.iqs = fft_iqs

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

    def prepare_train_data(self, idx):
        x = self.iqs[idx, :, :]
        x = np.expand_dims(x, axis=0)
        my = self.mod_labels[idx]

        inputs = dict(iqs=x)
        targets = dict(labels=my)
        input_metas = DC(dict(file_name=0), cpu_only=True)

        return dict(inputs=inputs, input_metas=input_metas, targets=targets)

    def prepare_test_data(self, idx):
        x = self.iqs[idx, :, :]
        x = np.expand_dims(x, axis=0)
        inputs = dict(iqs=x)
        input_metas = DC(dict(file_name=0), cpu_only=True)

        return dict(inputs=inputs, input_metas=input_metas)

    def __getitem__(self, idx):

        if self.test_mode:
            data = self.prepare_test_data(idx)
        else:
            data = self.prepare_train_data(idx)

        return data

    def evaluate(self, results, logger=None):
        results = format_results(results)
        eval_results = dict()

        def _eval(res_mod):
            y_ = np.zeros((1, self.num_mod), dtype=np.float32)
            sorted_anchor_index = np.argsort(res_mod)
            y_[0, sorted_anchor_index[-1]] = 1.0

            return y_

        y_ = list(map(_eval, results['Final']))
        y_ = np.concatenate(y_, axis=0)
        dy = self.gt - y_
        dy = np.sum(np.abs(dy), axis=1)
        for snr_index in range(self.num_snr):
            snr_left_index = self.snr_range[snr_index]
            snr_right_index = self.snr_range[snr_index + 1]
            accuracy = (snr_right_index - snr_left_index - np.count_nonzero(
                dy[snr_left_index:snr_right_index])) * 1.0 / (snr_right_index - snr_left_index)
            eval_results[f'snr_{self.snrs[snr_index]:02d}dB'] = accuracy
        eval_results['snr_mean_all'] = (len(self) - np.count_nonzero(dy)) * 1.0 / len(self)

        return eval_results


@DATASETS.register_module()
class CSSSDetSingleStageV2(BaseCSSS):
    def __init__(self, file_info, data_root=None, test_mode=False):
        super(CSSSDetSingleStageV2, self).__init__(file_info, data_root, test_mode)
        fft_iqs = np.zeros((len(self), 2, self.time_length), dtype=np.float32)
        mod_labels = np.ones(len(self) * self.num_band, dtype=np.int64) * self.num_mod
        band_labels = np.zeros((len(self), self.num_band), dtype=np.float32)
        self.gt = np.zeros((len(self), self.num_mod * self.num_band), dtype=np.float32)
        for idx in range(len(self)):
            iq = self.iqs[idx, 0, :] + 1j * self.iqs[idx, 1, :]
            iq = np.fft.fft(iq)
            iq = np.fft.fftshift(iq)
            iq = np.vstack((np.real(iq), np.imag(iq)))
            fft_iqs[idx, :, :] = iq
            band_labels[idx, self.band_labels[idx, 0]] = 1
            band_labels[idx, self.band_labels[idx, 1]] = 1
            mod_labels[idx * self.num_band + self.band_labels[idx, 0]] = self.mod_labels[idx, 0]
            mod_labels[idx * self.num_band + self.band_labels[idx, 1]] = self.mod_labels[idx, 1]
            for i in range(2):
                col_index = self.band_labels[idx, i] * self.num_mod + self.mod_labels[idx, i]
                self.gt[idx, col_index] = 1.0

        self.mod_labels = mod_labels
        self.band_labels = band_labels
        self.iqs = fft_iqs
        del self.iqs_

    def prepare_train_data(self, idx):
        x = self.iqs[idx * self.num_band:(idx + 1) * self.num_band, :, :]
        my = self.mod_labels[idx * self.num_band:(idx + 1) * self.num_band]

        inputs = dict(iqs=x)
        targets = dict(labels=my)
        input_metas = DC(dict(file_name=0), cpu_only=True)

        return dict(inputs=inputs, input_metas=input_metas, targets=targets)

    def prepare_test_data(self, idx):
        x = self.iqs[idx * self.num_band:(idx + 1) * self.num_band, :, :]
        inputs = dict(iqs=x)
        input_metas = DC(dict(file_name=0), cpu_only=True)

        return dict(inputs=inputs, input_metas=input_metas)

    def evaluate(self, results, logger=None):
        results = format_results(results)
        eval_results = dict()

        def _eval(res_mod):
            y_ = np.zeros((1, self.num_mod * self.num_band), dtype=np.float32)
            sorted_anchor_index = np.argsort(res_mod[:, -1])
            for anchor_index in sorted_anchor_index[:2]:
                y_[0, anchor_index * self.num_mod + np.argmax(res_mod[anchor_index, :-1])] = 1.0
            return y_

        y_ = list(map(_eval, results['Final']))
        y_ = np.concatenate(y_, axis=0)
        dy = self.gt - y_
        dy = np.sum(np.abs(dy), axis=1)
        for snr_index in range(self.num_snr):
            snr_left_index = self.snr_range[snr_index]
            snr_right_index = self.snr_range[snr_index + 1]
            accuracy = (snr_right_index - snr_left_index - np.count_nonzero(
                dy[snr_left_index:snr_right_index])) * 1.0 / (snr_right_index - snr_left_index)
            eval_results[f'acc_snr_{self.snrs[snr_index]:02d}dB'] = accuracy
        eval_results['acc_all_snr'] = (len(self) - np.count_nonzero(dy)) * 1.0 / len(self)

        y_ = np.reshape(y_, [-1, self.num_band, self.num_mod])
        y_ = np.sum(y_, axis=2)
        dy = self.band_labels - y_
        dy = np.sum(np.abs(dy), axis=1)
        for snr_index in range(self.num_snr):
            snr_left_index = self.snr_range[snr_index]
            snr_right_index = self.snr_range[snr_index + 1]
            accuracy = (snr_right_index - snr_left_index - np.count_nonzero(
                dy[snr_left_index:snr_right_index])) * 1.0 / (snr_right_index - snr_left_index)
            eval_results[f'band_acc_snr_{self.snrs[snr_index]:02d}dB'] = accuracy
        eval_results['band_acc_all_snr'] = (len(self) - np.count_nonzero(dy)) * 1.0 / len(self)

        return eval_results
