import os
import os.path as osp
import pickle

import matplotlib.pyplot as plt
import numpy as np
from cvxopt import solvers, matrix, div, mul
from scipy.special import softmax
from torch.utils.data import Dataset

from .builder import DATASETS
from ..common.fileio import dump as IODump
from ..common.fileio import load as IOLoad


def generate_targets(ann_file):
    annos = IOLoad(ann_file)
    anno_data = annos['data']
    mods_dict = annos['mods']
    targets = np.zeros((len(anno_data), len(mods_dict.keys())), dtype=np.float64)
    for item_index, item in enumerate(anno_data):
        targets[item_index, item['ann']['labels'][0]] = 1

    return targets.reshape(1, -1)


def ridge_regression(p, t, l):
    p = p.astype(dtype=np.float64)
    t = t.astype(dtype=np.float64)
    n = p.shape[0]
    G = -1 * np.eye(n, dtype=np.float64)
    h = np.zeros((n, 1), dtype=np.float64)
    P = np.dot(p, p.transpose()) + l / 2
    q = np.dot(p, t.transpose())
    A = np.ones((1, n), dtype=np.float64)
    b = np.ones((1, 1), dtype=np.float64)

    G = matrix(G)
    h = matrix(h)
    P = matrix(P)
    q = matrix(q)
    A = matrix(A)
    b = matrix(b)

    sol = solvers.qp(P, q, G, h, A=A, b=b)

    return np.array(sol['x'].T), float(sol['primal objective'])


def cross_entropy(p, t, l):
    # TODO: There are some bug in this function about div(), the matrix y has zero elements
    p = p.astype(dtype=np.float64)
    t = t.astype(dtype=np.float64)
    n = p.shape[0]
    G = -1 * np.eye(n, dtype=np.float64)
    h = np.zeros((n, 1), dtype=np.float64)
    A = np.ones((1, n), dtype=np.float64)
    b = np.ones((1, 1), dtype=np.float64)
    l = float(l)

    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)

    p = matrix(p)
    t = matrix(t)

    h_d = np.ones((n, 1), dtype=np.float64)
    h_d = matrix(h_d)

    def F(x=None, z=None):
        if x is None:
            x0 = np.ones((n, 1), dtype=np.float64) / n
            x0[0, 0] = 1 - np.sum(x0)
            return 0, matrix(x0)

        y = p.T * x
        f1 = matrix(-np.sum(np.log(np.power(np.array(y), np.array(t.T)))))
        # f1 = -t * log(y)
        f2 = x.T * x * 0.5 * l
        f = sum(f1 + f2)
        d1 = div(t.T, y)
        Df = -p * d1 + l * x
        if z is None:
            return f, Df.T
        d1 = div(d1, y)
        H = (mul(h_d * d1.T, p) * p.T + l) * z[0]

        return f, Df.T, H

    sol = solvers.cp(F, G=G, h=h, A=A, b=b)

    return np.array(sol['x'].T), float(sol['primal objective'])


@DATASETS.register_module()
class WTIMCDataset(Dataset):
    """Base dataset for modulation classification.

    The annotation format is shown as follows. The `ann` field is optional for
    testing.

        .. code-block:: none

            [
                data:[
                    {
                        'filename': '000000162059.npy',
                        'ann':{
                            'labels': list[int],
                            'snrs': list[float],
                        }

                    }
                    ...
                ]

                mods:{
                    'BPSK': 0, 
                    ...
                }

                snrs:{
                    -10: 0, 
                    ...
                }

                filters:[
                    [0.01, 0.005],
                    ...
                ]
            ]

    Args:
        ann_file (str): Annotation file path.
        iq (bool): If set True, the In-phase/Quadrature data (2, N) will be used as input
        ap (bool): If set True, the Amplitude/Phase data (2, N) will be used as input
        co (bool): If set True, the constellation data (H, W) will be used as input
        filter_config (int, optional): the idx to decide which kind of constellation data should be used. Default is 0.
        data_root (str, optional): Data root for ``ann_file``, ``In-phase/Quadrature data``, ``Amplitude/Phase data``,
                                   ``constellation data``.
        confusion_plot (bool, optional): If plot the confusion matrix
        multi_label (bool, optional): If set true, the input signal is the multi_label of different signals with
                                      different modulation and snr.
        test_mode (bool, optional): If set True, annotation will not be loaded.
        snr_threshold (float, optional): When the use_snr_lables is True, the item in dataset will be added extra info
                                         to indicate its snr property
        use_snr_label (bool, optional): It is worked with snr_threshold
        item_weights (list, optional): If the item is low snr, then the high head scale the gradient by a small value
        channel_mode (bool, optional): If set true, the original data shape (1, 2, N) is converted to (2, 1, N)
        use_hard_label (bool, optional): If set true, the modulations are grouped by easy and hard to predict separately.
        hard_modulations (list[str], optional): The hard modulations which can not be separated from other modulations.
        data_aug (bool, optional): If set true, make data augmentation for iq or ap (not include co data)
    """
    CLASSES = None
    SNRS = None

    def __init__(self, ann_file, iq=True, ap=False, co=False,
                 filter_config=0, data_root=None, confusion_plot=False,
                 multi_label=False, test_mode=False, snr_threshold=0,
                 use_snr_label=False, item_weights=None, channel_mode=False,
                 use_hard_label=False, hard_modulations=None, data_aug=False,
                 use_teacher_label=False, teacher_config=None):
        self.ann_file = ann_file
        self.iq = iq
        self.ap = ap
        self.co = co
        self.filter_config = filter_config
        self.data_root = data_root
        self.confusion_plot = confusion_plot
        self.multi_label = multi_label
        self.test_mode = test_mode
        self.snr_threshold = snr_threshold
        self.use_snr_label = use_snr_label
        self.item_weights = item_weights
        self.channel_mode = channel_mode
        self.use_hard_label = use_hard_label
        self.use_teacher_label = use_teacher_label
        self.teacher_config = teacher_config
        self.hard_modulations = hard_modulations
        self.data_aug = data_aug

        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)

        # load annotations
        self.data_infos, self.mods_dict, self.snrs_dict, self.filter = self.load_annotations(
            self.ann_file)
        self.index_class_dict = {index: mod for mod,
                                                index in self.mods_dict.items()}
        self.index_snr_dict = {index: float(snr)
                               for snr, index in self.snrs_dict.items()}

        self.CLASSES = [''] * len(self.index_class_dict)
        self.SNRS = [0.0] * len(self.index_snr_dict)

        for class_index in self.index_class_dict.keys():
            self.CLASSES[class_index] = self.index_class_dict[class_index]
        for snr_index in self.index_snr_dict.keys():
            self.SNRS[snr_index] = self.index_snr_dict[snr_index]

        #
        if self.use_teacher_label:
            self.teacher_labels = self._generate_teacher_labels()

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        if self.multi_label:
            self._generate_mod_labels()

        # the targets are for knowledge distillation
        self.targets = generate_targets(self.ann_file)

    def _set_group_flag(self):
        """Set flag according to the mean value (ms) of signal data snrs.

        set as group i, where self.SNRS[i]<=ms<self.SNRS[i+1].
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        # for idx in range(len(self)):
        #     ann = self.get_ann_info(idx)
        #     snrs = ann['snrs']
        #     mean_snr = np.mean(snrs)
        #     self.flag[idx] = np.where(np.array(self.SNRS) <= mean_snr)[0][-1]

    def _generate_mod_labels(self):
        self.mix_mod_labels = np.zeros(
            (len(self), len(self.CLASSES)), dtype=np.float32)
        for idx in range(len(self)):
            ann = self.get_ann_info(idx)
            labels = ann['mod_labels']
            for class_index in labels:
                self.mix_mod_labels[idx, class_index] = 1

    def _generate_teacher_labels(self):
        data_dict = pickle.load(open(os.path.join(self.data_root, self.teacher_config['filename']), 'rb'))
        teacher_labels = data_dict[self.teacher_config['lambda_val']]

        return teacher_labels.astype(dtype=np.float32)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        annos = IOLoad(ann_file)
        anno_data = annos['data']
        mods_dict = annos['mods']
        snrs_dict = annos['snrs']
        filter_config = annos['filters'][self.filter_config]
        if self.use_snr_label:
            tmp_anno_data = []
            for item in anno_data:
                if item['ann']['snrs'][0] >= self.snr_threshold:
                    item['ann']['snr_labels'] = [0]
                    if self.item_weights is not None:
                        item['ann']['low_weight'] = [self.item_weights[0]]
                        item['ann']['high_weight'] = [self.item_weights[1]]
                else:
                    item['ann']['snr_labels'] = [1]
                    if self.item_weights is not None:
                        item['ann']['low_weight'] = [self.item_weights[1]]
                        item['ann']['high_weight'] = [self.item_weights[0]]
                tmp_anno_data.append(item)
            anno_data = tmp_anno_data
        rmods_dict = {v: k for k, v in mods_dict.items()}
        if self.use_hard_label:
            tmp_anno_data = []
            for item in anno_data:
                if rmods_dict[item['ann']['labels'][0]] in self.hard_modulations:
                    item['ann']['hard_labels'] = [1]
                else:
                    item['ann']['hard_labels'] = [0]
                tmp_anno_data.append(item)
            anno_data = tmp_anno_data

        return anno_data, mods_dict, snrs_dict, filter_config

    def __getitem__(self, idx):

        if self.test_mode:
            data = self.prepare_test_data(idx)
        else:
            data = self.prepare_train_data(idx)

        return data

    def get_ann_info(self, idx):
        """Get annotation by idx.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified idx.
        """
        ann_info_keys = []
        ann_info_vals = []

        mod_labels = self.data_infos[idx]['ann']['labels']
        ann_info_keys.append('mod_labels')
        ann_info_vals.append(mod_labels)

        snrs = self.data_infos[idx]['ann']['snrs']
        ann_info_keys.append('snrs')
        ann_info_vals.append(snrs)

        if self.use_snr_label:
            snr_labels = self.data_infos[idx]['ann']['snr_labels']
            ann_info_keys.append('snr_labels')
            ann_info_vals.append(snr_labels)

            if self.item_weights is not None:
                low_weight = self.data_infos[idx]['ann']['low_weight']
                ann_info_keys.append('low_weight')
                ann_info_vals.append(low_weight)

                high_weight = self.data_infos[idx]['ann']['high_weight']
                ann_info_keys.append('high_weight')
                ann_info_vals.append(high_weight)

        if self.use_hard_label:
            hard_labels = self.data_infos[idx]['ann']['hard_labels']
            ann_info_keys.append('hard_labels')
            ann_info_vals.append(hard_labels)

        return self._parse_ann_info(ann_info_keys, ann_info_vals)

    def get_cat_ids(self, idx):
        """Get category ids by idx.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the data of specified idx.
        """

        return self.data_infos[idx]['ann']['labels']

    @staticmethod
    def _parse_ann_info(ann_info_keys, ann_info_vals):
        """Parse labels and snrs annotation

        Args:
            ann_info_keys (list[str]): ann name of an item
            ann_info_vals (list[float]): ann val of an item

        Returns:
            dict: A dict
        """
        ann = dict()
        for i, key_name in enumerate(ann_info_keys):
            if 'label' in key_name:
                ann[key_name] = np.array(ann_info_vals[i], dtype=np.int64)
            else:
                ann[key_name] = np.array(ann_info_vals[i], dtype=np.float32)

        return ann

    def load_seq_data(self, data_type, file_name):
        file_path = osp.join(
            self.data_root, 'sequence_data', data_type, file_name)
        seq_data = np.load(file_path)
        seq_data = seq_data.astype(np.float32)
        num_path = 2
        if self.data_aug:
            seq_aug = np.roll(seq_data, -1) - seq_data
            seq_data = np.vstack([seq_data, seq_aug])
            num_path += 2
        if self.channel_mode:
            seq_data = np.reshape(seq_data, (num_path, 1, -1))
        else:
            seq_data = np.reshape(seq_data, (1, num_path, -1))

        return seq_data

    def prepare_train_data(self, idx):
        """Get training data.

        Args.
         idx (int): Index of data

        Returns:
            dict: Training data and annotation
        """
        item_info = self.data_infos[idx]

        file_name = item_info['filename']

        data = {}

        if self.iq:
            iq_data = self.load_seq_data('iq', file_name)
            data['iqs'] = iq_data

        if self.ap:
            ap_data = self.load_seq_data('ap', file_name)
            data['aps'] = ap_data

        if self.co:
            co_path = osp.join(self.data_root, 'constellation_data',
                               'filter_size_{:<.3f}_stride_{:<.3f}'.format(self.filter[0], self.filter[1]), file_name)
            co_data = np.load(co_path)
            co_data = co_data.astype(np.float32)
            co_data = np.expand_dims(co_data, axis=0)
            data['cos'] = co_data

        ann = self.get_ann_info(idx)
        data['mod_labels'] = ann['mod_labels']
        if self.use_snr_label:
            data['snr_labels'] = ann['snr_labels']

        if self.item_weights is not None:
            data['low_weight'] = ann['low_weight']
            data['high_weight'] = ann['high_weight']

        if self.use_hard_label:
            data['hard_labels'] = ann['hard_labels']

        if self.use_teacher_label:
            data['teacher_labels'] = self.teacher_labels[idx, :]

        return data

    def prepare_test_data(self, idx):
        """Get testing data.

        Args.
         idx (int): Index of data

        Returns:
            dict: Testing data and annotation
        """
        item_info = self.data_infos[idx]

        file_name = item_info['filename']

        data = {}

        if self.iq:
            iq_data = self.load_seq_data('iq', file_name)
            data['iqs'] = iq_data

        if self.ap:
            ap_data = self.load_seq_data('ap', file_name)
            data['aps'] = ap_data

        if self.co:
            co_path = osp.join(self.data_root, 'constellation_data',
                               'filter_size_{:<.3f}_stride_{:<.3f}'.format(self.filter[0], self.filter[1]), file_name)
            co_data = np.load(co_path)
            co_data = co_data.astype(np.float32)
            co_data = np.expand_dims(co_data, axis=0)
            data['cos'] = co_data

        return data

    @staticmethod
    def _plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
        fig = plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        return fig

    def _generate_accuracy_upperbound(self, results_list, prefix=''):
        confusion_matrix = np.zeros((len(self.SNRS), len(
            self.CLASSES), len(self.CLASSES)), dtype=np.float64)

        for idx in range(len(self)):
            ann = self.get_ann_info(idx)
            snrs = ann['snrs']
            labels = ann['mod_labels']
            if snrs.size == 1 and labels.size == 1:
                predict_class_indexes = []
                for results in results_list:
                    predict_class_index = int(np.argmax(results[idx, :]))
                    predict_class_indexes.append(predict_class_index)
                if int(labels[0]) in predict_class_indexes:
                    confusion_matrix[self.snrs_dict['{:.3f}'.format(snrs[0])],
                                     labels[0], labels[0]] += 1
                else:
                    confusion_matrix[self.snrs_dict['{:.3f}'.format(snrs[0])],
                                     labels[0], predict_class_indexes[0]] += 1
            else:
                raise ValueError('Please check your dataset, the size of snrs and labels are both 1 for any item. '
                                 'However, the current item with the idx {:d} has the snrs size {:d} and the '
                                 'labels size {:d}'.format(idx, snrs.size, labels.size))

        confusion_matrix = confusion_matrix / \
                           np.expand_dims(np.sum(confusion_matrix, axis=2), axis=2)

        eval_results = dict()
        for snr_index, snr in enumerate(self.SNRS):
            conf = confusion_matrix[snr_index, :, :]
            cor = np.sum(np.diag(conf))
            ncor = np.sum(conf) - cor
            eval_results[prefix +
                         'snr_{:.3f}'.format(snr)] = 1.0 * cor / (cor + ncor)
            if self.confusion_plot:
                eval_results[prefix + 'snr_{:.3f}_conf_figure'.format(snr)] = self._plot_confusion_matrix(
                    conf, title='Confusion matrix (SNR={:.3f})'.format(snr), labels=self.CLASSES)
        conf = np.sum(confusion_matrix, axis=0) / len(self.SNRS)
        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor
        eval_results[prefix + 'snr_mean_all_upperbound'] = 1.0 * cor / (cor + ncor)
        if self.confusion_plot:
            eval_results[prefix + 'snr_mean_all_conf_figure'] = self._plot_confusion_matrix(
                conf, title='Mean of all snr Confusion matrix', labels=self.CLASSES)
        return eval_results

    def _evaluate_mod(self, results, prefix=''):
        confusion_matrix = np.zeros((len(self.SNRS), len(
            self.CLASSES), len(self.CLASSES)), dtype=np.float64)

        for idx in range(len(self)):
            ann = self.get_ann_info(idx)
            snrs = ann['snrs']
            labels = ann['mod_labels']
            if snrs.size == 1 and labels.size == 1:
                predict_class_index = int(np.argmax(results[idx, :]))
                confusion_matrix[self.snrs_dict['{:.3f}'.format(snrs[0])],
                                 labels[0], predict_class_index] += 1
            else:
                raise ValueError('Please check your dataset, the size of snrs and labels are both 1 for any item. '
                                 'However, the current item with the idx {:d} has the snrs size {:d} and the '
                                 'labels size {:d}'.format(idx, snrs.size, labels.size))

        confusion_matrix = confusion_matrix / \
                           np.expand_dims(np.sum(confusion_matrix, axis=2), axis=2)

        eval_results = dict()
        for snr_index, snr in enumerate(self.SNRS):
            conf = confusion_matrix[snr_index, :, :]
            cor = np.sum(np.diag(conf))
            ncor = np.sum(conf) - cor
            eval_results[prefix +
                         'snr_{:.3f}'.format(snr)] = 1.0 * cor / (cor + ncor)
            if self.confusion_plot:
                eval_results[prefix + 'snr_{:.3f}_conf_figure'.format(snr)] = self._plot_confusion_matrix(
                    conf, title='Confusion matrix (SNR={:.3f})'.format(snr), labels=self.CLASSES)
        conf = np.sum(confusion_matrix, axis=0) / len(self.SNRS)
        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor
        eval_results[prefix + 'snr_mean_all'] = 1.0 * cor / (cor + ncor)
        if self.confusion_plot:
            eval_results[prefix + 'snr_mean_all_conf_figure'] = self._plot_confusion_matrix(
                conf, title='Mean of all snr Confusion matrix', labels=self.CLASSES)
        return eval_results

    def _evaluate_mod_mix(self, results, prefix=''):

        def calculate_the_best_classification_threshold(predictions, gts):
            ascending_index = np.argsort(predictions)
            predictions = predictions[ascending_index]
            gts = gts[ascending_index]

            classification_accuracy = -1
            best_threshold_index = -1
            for idx in range(len(self) + 1):
                negative_items = gts[0:idx]
                positive_items = gts[idx:len(self)]

                correct_number = (negative_items == 0).sum() + \
                                 (positive_items == 1).sum
                current_classification_accuracy = correct_number / len(self)
                if current_classification_accuracy >= classification_accuracy:
                    classification_accuracy = current_classification_accuracy
                    best_threshold_index = idx

            if best_threshold_index == len(self):
                best_threshold = predictions[len(
                    self) - 1] + np.finfo(np.float32).eps
            else:
                best_threshold = predictions[best_threshold_index]

            return best_threshold, classification_accuracy

        best_thresholds = np.zeros(len(self), dtype=np.float32)
        classification_accuracys = np.zeros(len(self), dtype=np.float32)
        for class_ids in range(len(self.CLASSES)):
            best_thresholds[class_ids], classification_accuracys[
                class_ids] = calculate_the_best_classification_threshold(results[:, class_ids],
                                                                         self.mix_mod_labels[:, class_ids])

        eval_results = dict()
        for class_ids, class_name in enumerate(self.CLASSES):
            eval_results[prefix + '{}_thr'.format(
                class_name)] = best_thresholds[class_ids]
            eval_results[prefix + '{}_acc'.format(
                class_name)] = classification_accuracys[class_ids]

        return eval_results

    def _evaluate_snr(self, results, prefix):
        confusion_matrix = np.zeros((len(self.SNRS), 2), dtype=np.float64)

        for idx in range(len(self)):
            ann = self.get_ann_info(idx)
            snrs = ann['snrs']
            labels = ann['snr_labels']
            if snrs.size == 1 and labels.size == 1:
                predict_class_index = int(np.argmax(results[idx, :]))
                row_index = self.snrs_dict['{:.3f}'.format(snrs[0])]
                if int(labels[0]) == predict_class_index:
                    confusion_matrix[row_index, 0] += 1
                else:
                    confusion_matrix[row_index, 1] += 1
            else:
                raise ValueError('Please check your dataset, the size of snrs and labels are both 1 for any item. '
                                 'However, the current item with the idx {:d} has the snrs size {:d} and the '
                                 'labels size {:d}'.format(idx, snrs.size, labels.size))

        confusion_matrix = confusion_matrix / \
                           np.expand_dims(np.sum(confusion_matrix, axis=1), axis=1)

        eval_results = dict()
        for snr_index, snr in enumerate(self.SNRS):
            eval_results[prefix + 'snr_{:.3f}_acc'.format(
                snr)] = confusion_matrix[snr_index, 0]
        conf = np.sum(confusion_matrix, axis=0) / len(self.SNRS)
        eval_results[prefix + 'snr_acc'] = conf[0]

        return eval_results

    def _evaluate_hard(self, results, prefix):
        confusion_matrix = np.zeros((len(self.CLASSES), 2), dtype=np.float64)

        for idx in range(len(self)):
            ann = self.get_ann_info(idx)
            mod_labels = ann['mod_labels']
            hard_labels = ann['hard_labels']
            if mod_labels.size == 1 and hard_labels.size == 1:
                predict_class_index = int(np.argmax(results[idx, :]))
                row_index = mod_labels[0]
                if int(hard_labels[0]) == predict_class_index:
                    confusion_matrix[row_index, 0] += 1
                else:
                    confusion_matrix[row_index, 1] += 1
            else:
                raise ValueError(
                    'Please check your dataset, the size of mod_labels and hard_labels are both 1 for any item. '
                    'However, the current item with the idx {:d} has the mod_labels size {:d} and the '
                    'hard_labels size {:d}'.format(idx, mod_labels.size, hard_labels.size))

        confusion_matrix = confusion_matrix / \
                           np.expand_dims(np.sum(confusion_matrix, axis=1), axis=1)

        eval_results = dict()
        for mod_index, mod in enumerate(self.CLASSES):
            eval_results[prefix + '{}_acc'.format(mod)] = confusion_matrix[int(self.mods_dict[mod]), 0]
        conf = np.sum(confusion_matrix, axis=0) / len(self.CLASSES)
        eval_results[prefix + 'mod_acc'] = conf[0]

        return eval_results

    def _reshape_result(self, results, num_cols):
        results = [result.reshape(1, num_cols) for result in results]
        results = np.concatenate(results, axis=0)
        results = np.reshape(results, (len(self), -1))
        return results

    def process_single_head(self, results, prefix=''):

        if 'snr' in prefix:
            results = self._reshape_result(results, 2)
            eval_results = self._evaluate_snr(results, prefix=prefix)
        elif 'hard' in prefix:
            results = self._reshape_result(results, 2)
            eval_results = self._evaluate_hard(results, prefix=prefix)
        else:
            results = self._reshape_result(results, len(self.CLASSES))
            if not self.multi_label:
                eval_results = self._evaluate_mod(results, prefix=prefix)
            else:
                eval_results = self._evaluate_mod_mix(results, prefix=prefix)

        return eval_results

    def _process_upperbound(self, results_dict, prefix=''):
        new_results = []
        for key_str in results_dict.keys():
            results = self._reshape_result(results_dict[key_str], len(self.CLASSES))
            new_results.append(results)

        eval_results = self._generate_accuracy_upperbound(new_results)

        return eval_results

    def _process_weight(self, results_dict, prefix=''):
        pre_matrix = []
        for key_str in results_dict.keys():
            pre_data = results_dict[key_str]
            pre_data = self._reshape_result(pre_data, len(self.CLASSES))
            pre_data = softmax(pre_data, axis=1)
            pre_matrix.append(pre_data.reshape(1, -1))
        pre_matrix = np.vstack(pre_matrix)

        r_w, r_o = ridge_regression(pre_matrix, self.targets, 0)
        c_w, c_o = cross_entropy(pre_matrix, self.targets, 0)

        r_eval_results = self._evaluate_mod(np.reshape(np.dot(r_w, pre_matrix), (-1, len(self.CLASSES))),
                                            prefix='rg-weight_')
        c_eval_results = self._evaluate_mod(np.reshape(np.dot(c_w, pre_matrix), (-1, len(self.CLASSES))),
                                            prefix='ce-weight_')
        return r_eval_results, c_eval_results

    def evaluate(self, results, logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        """
        if isinstance(results[0], dict):
            format_results = {key_str: [] for key_str in results[0].keys()}
            for item in results:
                for key_str in item.keys():
                    format_results[key_str].append(item[key_str])

            eval_results = dict()
            for key_str in format_results.keys():
                sub_eval_results = self.process_single_head(
                    format_results[key_str], prefix=key_str + '/')
                eval_results.update(sub_eval_results)

            format_results.pop('snr', None)
            format_results.pop('hard', None)
            if len(format_results.keys()) > 1:
                eval_results_upperbound = self._process_upperbound(format_results)
                # r_eval_results, c_eval_results = self._process_weight(format_results)
                eval_results.update(eval_results_upperbound)
                # eval_results.update(r_eval_results)
                # eval_results.update(c_eval_results)
        else:
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

        def format_out_single(results, save_path):
            if 'snr' in save_path:
                if self.item_weights is None:
                    results = [result.reshape(1, 2)
                               for result in results]
                else:
                    results = [result.reshape(1, len(self.item_weights))
                               for result in results]
                results = np.concatenate(results, axis=0)
                results = np.reshape(results, (len(self), -1))
            else:
                results = [result.reshape(1, len(self.CLASSES))
                           for result in results]
                results = np.concatenate(results, axis=0)
                results = np.reshape(results, (len(self), -1))
            np.save(save_path, results)

        if self.use_snr_label:
            format_results = {key_str: [] for key_str in results[0].keys()}
            for item in results:
                for key_str in item.keys():
                    format_results[key_str].append(item[key_str])

            for key_str in format_results.keys():
                sub_results = format_results[key_str]
                save_path = os.path.join(out_dir, key_str + '_pre.npy')
                format_out_single(sub_results, save_path)
        else:
            save_path = os.path.join(out_dir, 'pre.npy')
            format_out_single(results, save_path)

        json_results = dict(CLASSES=self.CLASSES, SNRS=self.SNRS,
                            mods_dict=self.mods_dict, snrs_dict=self.snrs_dict, ANN=[])
        for idx in range(len(self)):
            ann = self.get_ann_info(idx)
            json_results['ANN'].append(ann)

        IODump(json_results, osp.join(out_dir, 'ann.json'))
