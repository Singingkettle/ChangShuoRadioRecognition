import os.path as osp
import pickle

import h5py
import numpy as np
import tqdm
import zlib
from scipy.special import expit, softmax
from sklearn.metrics import precision_recall_curve
from torch.utils.data import Dataset

from .builder import DATASETS
from .merge.methods import get_merge_weight_by_grid_search
from .utils import format_results, reshape_results, Constellation


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

    def __init__(self, file_name, grid_step=0.09, data_root=None, test_mode=False):

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
        self.num = self.X.shape[0]
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        del data

    def __len__(self):
        return self.num

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

        mod_labels = np.array(y[0], dtype=np.int64)
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
            pre_data = softmax(pre_data, axis=1)
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


def _det_label(y):
    channel_labels = np.zeros(24, dtype=np.float32)
    mod_labels = np.zeros(24 * 13, dtype=np.float32)
    for i, val in enumerate(y):
        if val > 0:
            channel_labels[i] = 1
            mod_labels[i * 13 + val - 1] = 1

    return dict(mod_labels=mod_labels, channel_labels=channel_labels)


def _bce_label(y):
    mod_labels = np.zeros(24 * 13, dtype=np.float32)
    for i, val in enumerate(y):
        if val > 0:
            mod_labels[i * 13 + val - 1] = 1.0

    return dict(mod_labels=mod_labels)


def _ind_label(y):
    mod_labels = np.zeros(13, dtype=np.float32)
    channel_labels = np.zeros(24, dtype=np.float32)
    order_labels = np.zeros(4, dtype=np.float32)
    l = []
    for i, val in enumerate(y):
        if val > 0:
            mod_labels[val - 1] = 1
            channel_labels[i] = 1
            l.append(val)

    if len(l) > 1:
        if l[0] > l[1]:
            order_labels[0] = 1
        elif l[0] < l[1]:
            order_labels[1] = 1
        else:
            order_labels[2] = 1
    else:
        order_labels[3] = 1

    return dict(mod_labels=mod_labels, channel_labels=channel_labels, order_labels=order_labels)


TARGET = dict(bce=_bce_label, det=_det_label, ind=_ind_label)


def _bce_eval(results, mod_threshold=0.1):
    results = results['Final']
    y_ = np.zeros((len(results), 24), dtype=np.int64)
    for i, res in enumerate(results):
        res = np.reshape(res, [24, 13])
        max_scores = np.max(res, axis=1)
        max_scores_index = np.argmax(res, axis=1)
        sorted_index = np.argsort(max_scores)
        selected_number = 0
        for index in sorted_index[-2:]:
            if max_scores[index] >= mod_threshold:
                y_[i, index] = max_scores_index[index] + 1
                selected_number += 1

        if selected_number == 0:
            y_[i, sorted_index[-1]] = max_scores_index[sorted_index[-1]] + 1

    return y_


def _det_eval(results, channel_threshold=0.1, mod_threshold=0.1):
    channel_results = [value for key, value in results.items() if 'channel' in key.lower()][0]
    mod_results = [value for key, value in results.items() if 'mod' in key.lower()][0]
    mod_results = [np.reshape(res, [24, 13]) for res in mod_results]

    y_ = np.zeros((len(channel_results), 24), dtype=np.int64)
    for i, cs in enumerate(channel_results):
        sorted_c_index = np.argsort(cs)
        selected_number = 0
        for c_index in sorted_c_index[-2:]:
            if cs[c_index] >= channel_threshold:
                max_m_score = np.max(mod_results[i][c_index, :])
                max_m_index = np.argmax(mod_results[i][c_index, :])
                if max_m_score >= mod_threshold:
                    y_[i, c_index] = max_m_index + 1
                    selected_number += 1

        if selected_number == 0:
            c_index = sorted_c_index[-1]
            max_m_index = np.argmax(mod_results[i][c_index, :])
            y_[i, c_index] = max_m_index + 1

    return y_


def _ind_eval(results):
    channel_results = [value for key, value in results.items() if 'channel' in key.lower()][0]
    mod_results = [value for key, value in results.items() if 'mod' in key.lower()][0]
    order_results = [value for key, value in results.items() if 'order' in key.lower()][0]
    y_ = np.zeros((len(channel_results), 24), dtype=np.int64)
    for i, order_scores in enumerate(order_results):
        max_os_index = np.argmax(order_scores)
        sorted_cs_index = np.argsort(channel_results[i])
        sorted_ms_index = np.argsort(mod_results[i])
        if max_os_index == 3:
            y_[i, sorted_cs_index[-1]] = sorted_ms_index[-1] + 1
        elif max_os_index == 2:
            y_[i, sorted_cs_index[-1]] = sorted_ms_index[-1] + 1
            y_[i, sorted_cs_index[-2]] = sorted_ms_index[-1] + 1
        else:
            sorted_cs_index = np.sort(sorted_cs_index[-2:])
            sorted_ms_index = np.sort(sorted_ms_index[-2:])
            if max_os_index == 1:
                y_[i, sorted_cs_index[-1]] = sorted_ms_index[-1] + 1
                y_[i, sorted_cs_index[-2]] = sorted_ms_index[-2] + 1
            else:
                y_[i, sorted_cs_index[-1]] = sorted_ms_index[-2] + 1
                y_[i, sorted_cs_index[-2]] = sorted_ms_index[-1] + 1

    return y_


EVALUATE = dict(bce=_bce_eval, det=_det_eval, ind=_ind_eval)


@DATASETS.register_module()
class GBSenseAdvanced(GBSenseBasic):
    def __init__(self, file_name, grid_step=0.01, data_root=None, head=None, test_mode=False, is_con=False):
        super(GBSenseAdvanced, self).__init__(file_name, grid_step, data_root, test_mode)
        self.target_handle = TARGET[head['type']]
        self.eval_handle = EVALUATE[head['type']]
        self.type = head['type']
        self.is_con = is_con
        if self.is_con:
            self.convert_tool = Constellation([0.02], [0.02])

            data = []
            if self.test_mode:
                co_file_name = 'test_co.pkl'
            else:
                co_file_name = 'train_co.pkl'
            if osp.isfile(osp.join(data_root, co_file_name)):
                self.X = pickle.load(open(osp.join(data_root, co_file_name), 'rb'))
            else:
                for idx in tqdm.tqdm(range(len(self))):
                    x = self.X[idx, :, :]
                    x = np.reshape(x, [-1, 2])
                    x = x[:, 0] + 1j * x[:, 1]
                    x = x / np.sum((np.abs(x)))
                    x = np.vstack((np.real(x), np.imag(x)))
                    cos, _ = self.convert_tool.generate_by_filter(x)
                    cos = cos[0].astype(np.float32)
                    cos = np.expand_dims(cos, axis=0)
                    cdata = zlib.compress(cos.tobytes())
                    data.append(cdata)
                    pickle.dump(data, open(osp.join(data_root, co_file_name), 'wb'), protocol=4)

                self.X = data

        if 'ind' != self.type:
            self.eval_kwargs = head.pop('type')

    def prepare_test_data(self, idx):
        if self.is_con:
            x = self.X[idx]
            x = zlib.decompress(x)
            x = np.frombuffer(x, dtype=np.float32).reshape([1, 100, 100])
            x = x.astype(np.float32)
            data = dict(cos=x)
        else:
            x = self.X[idx, :, :]
            x = np.reshape(x, [-1, 2])
            x = x[:, 0] + 1j * x[:, 1]
            x = x / np.sum((np.abs(x)))
            x = np.vstack((np.real(x), np.imag(x)))
            x = np.transpose(x)
            x = np.expand_dims(x, axis=1)
            x = x.astype(np.float32)
            data = dict(iqs=x)
        return data

    def prepare_train_data(self, idx):
        if self.is_con:
            x = self.X[idx]
            x = zlib.decompress(x)
            x = np.frombuffer(x, dtype=np.float32).reshape([1, 100, 100])
            x = x.astype(np.float32)
            data = dict(cos=x)
        else:
            x = self.X[idx, :, :]
            x = np.reshape(x, [-1, 2])
            x = x[:, 0] + 1j * x[:, 1]
            x = x / np.sum((np.abs(x)))
            x = np.vstack((np.real(x), np.imag(x)))
            x = np.transpose(x)
            x = np.expand_dims(x, axis=1)
            x = x.astype(np.float32)
            data = dict(iqs=x)
        y = self.Y[idx, :]
        target = self.target_handle(y)
        data.update(target)

        return data

    def __getitem__(self, idx):

        if self.test_mode:
            data = self.prepare_test_data(idx)
        else:
            data = self.prepare_train_data(idx)

        return data

    def get_eval_matrix(self, y_):
        eval_matrix = np.zeros((24, 13, 5), dtype=np.float64)

        success_pred_number = 0
        for idx in range(y_.shape[0]):
            preds = y_[idx, :]
            preds_index = np.nonzero(preds)[0]
            preds_index = preds_index.tolist()
            gts = self.Y[idx, :]
            gts_index = np.nonzero(gts)[0]
            gts_index = gts_index.tolist()

            gts_index_has_checked = []
            is_success = True
            for c_index in preds_index:
                if c_index in gts_index:
                    gts_index_has_checked.append(c_index)
                    if preds[c_index] == gts[c_index]:
                        eval_matrix[c_index, preds[c_index] - 1, 0] += 1
                    else:
                        eval_matrix[c_index, preds[c_index] - 1, 1] += 1
                        eval_matrix[c_index, gts[c_index] - 1, 2] += 1
                        is_success = False
                else:
                    eval_matrix[c_index, preds[c_index] - 1, 3] += 1
                    is_success = False

            for c_index in gts_index:
                if c_index not in gts_index_has_checked:
                    eval_matrix[c_index, gts[c_index] - 1, 4] += 1
                    is_success = False

            if is_success:
                success_pred_number += 1

        accuracy_by_item = 1.0 * success_pred_number / y_.shape[0]
        f1_by_channel = 2 * np.sum(eval_matrix[:, :, 0]) / (np.sum(eval_matrix[:, :, :]) + np.sum(eval_matrix[:, :, 0]))

        return accuracy_by_item, f1_by_channel

    def evaluate(self, results, logger=None):
        results = format_results(results)

        if self.type == 'bce':
            search_range = np.linspace(0.01, 1, 10, endpoint=False)
            search_range = [{'mod_threshold': val} for val in search_range[::-1]]
        elif self.type == 'det':
            c_search_range = np.linspace(0.01, 1, 10, endpoint=False)
            m_search_range = np.linspace(0.01, 1, 10, endpoint=False)
            search_range = []
            for c_val in c_search_range[::-1]:
                for m_val in m_search_range[::-1]:
                    search_range.append(dict(channel_threshold=c_val, mod_threshold=m_val))
        else:
            search_range = []

        if len(search_range) == 0:
            y_ = self.eval_handle(results)
            accuracy, f1 = self.get_eval_matrix(y_)
        else:
            y_ = self.eval_handle(results, **search_range[0])
            max_acc, max_f1 = self.get_eval_matrix(y_)
            for i in tqdm.tqdm(range(1, len(search_range))):
                y_ = self.eval_handle(results, **search_range[i])
                cur_acc, cur_f1 = self.get_eval_matrix(y_)
                if cur_acc >= max_acc:
                    max_acc = cur_acc
                    max_f1 = cur_f1

            accuracy = max_acc
            f1 = max_f1

        eval_results = dict()

        eval_results['Accuracy'] = accuracy
        eval_results['F1'] = f1

        return eval_results


@DATASETS.register_module()
class GBSenseAdvancedChannel(Dataset):
    CLASSES = ['ns', 'hs']

    def __init__(self, file_name, data_root=None, test_mode=False):
        
        self.test_mode = test_mode
        self.data_root = data_root
        self.y = pickle.load(open(osp.join(data_root, file_name), 'rb'))
        if not self.test_mode:
            self._set_group_flag()

    def __len__(self):
        return len(self.y)

    def _set_group_flag(self):
        """Set flag according to the mean value (ms) of signal data snrs.

        set as group i, where self.SNRS[i]<=ms<self.SNRS[i+1].
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def prepare_test_data(self, idx):
        x = np.load(osp.join(self.data_root, 'shift/channel', f'test_{idx:07d}.npy'))
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=1)
        data = dict(iqs=x)
        return data

    def prepare_train_data(self, idx):
        x = np.load(osp.join(self.data_root, 'shift/channel', f'train_{idx:07d}.npy'))
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=1)

        data = dict(iqs=x)
        y = np.zeros(1, dtype=np.float32)
        y[0] = self.y[idx]
        target = {'mod_labels': y}
        data.update(target)

        return data

    def __getitem__(self, idx):

        if self.test_mode:
            data = self.prepare_test_data(idx)
        else:
            data = self.prepare_train_data(idx)

        return data

    def evaluate(self, results, logger=None):
        results = format_results(results)
        precision, recall, thresholds = precision_recall_curve(self.y, results['Final'])
        # convert to f score
        fscore = (2 * precision * recall) / (precision + recall)
        fscore = np.nan_to_num(fscore)
        # locate the index of the largest f score
        ix = np.argmax(fscore)
        print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
        eval_results = {'FS': fscore[ix]}

        return eval_results


@DATASETS.register_module()
class GBSenseAdvancedMod(Dataset):
    CLASSES = ["APSK16", "APSK32", "APSK64", "ASK8", "BPSK", "OQPSK",
               "PSK16", "PSK8", "QAM128", "QAM16", "QAM256", "QAM64", "QPSK"]

    def __init__(self, file_name, data_root=None, test_mode=False):

        self.test_mode = test_mode
        self.data_root = data_root
        self.y = pickle.load(open(osp.join(data_root, file_name), 'rb'))
        if not self.test_mode:
            self._set_group_flag()

    def __len__(self):
        return len(self.y)

    def _set_group_flag(self):
        """Set flag according to the mean value (ms) of signal data snrs.

        set as group i, where self.SNRS[i]<=ms<self.SNRS[i+1].
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def prepare_test_data(self, idx):
        x = np.load(osp.join(self.data_root, 'shift/mod', f'test_{idx:07d}.npy'))
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=1)
        data = dict(iqs=x)
        return data

    def prepare_train_data(self, idx):
        x = np.load(osp.join(self.data_root, 'shift/mod', f'train_{idx:07d}.npy'))
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=1)

        data = dict(iqs=x)
        y = np.array(self.y[idx], dtype=np.int64)
        target = {'mod_labels': y}
        data.update(target)

        return data

    def __getitem__(self, idx):

        if self.test_mode:
            data = self.prepare_test_data(idx)
        else:
            data = self.prepare_train_data(idx)

        return data

    def get_confusion_matrix(self, results):
        confusion_matrix = np.zeros((len(self.CLASSES), len(self.CLASSES)), dtype=np.float64)
        for idx in range(len(results)):
            predict_label = int(np.argmax(results[idx]))
            gt_label = self.y[idx]
            confusion_matrix[gt_label, predict_label] += 1
        return confusion_matrix

    def evaluate(self, results, logger=None):
        results = format_results(results)
        conf = self.get_confusion_matrix(results['Final'])
        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor
        accuracy = 1.0 * cor / (cor + ncor)
        eval_results = {'ACC': accuracy}

        return eval_results