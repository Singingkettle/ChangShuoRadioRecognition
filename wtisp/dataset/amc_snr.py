import os.path as osp

import numpy as np
from torch.utils.data import Dataset

from .builder import DATASETS
from ..common.fileio import dump as IODump
from ..common.fileio import load as IOLoad


@DATASETS.register_module()
class WTISNRDataset(Dataset):
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
        data_root (str, optional): Data root for ``ann_file``, ``In-phase/Quadrature data``, 
        ``Amplitude/Phase data``, ``constellation data``.
        filter_config (int, optional): the idx to decide which kind of constellation data 
        should be used. Default is 0.
        multi_label (bool, optional): If set true, the input signal is the multi_label of different signals with
        different modulation and snr.
        test_mode (bool, optional): If set True, annotation will not be loaded.
    """
    CLASSES = None
    SNRS = None

    def __init__(self, ann_file, iq=True, ap=False, co=False,
                 filter_config=0, data_root=None, multi_label=False, test_mode=False):

        self.ann_file = ann_file
        self.iq = iq
        self.ap = ap
        self.co = co
        self.filter_config = filter_config
        self.data_root = data_root
        self.multi_label = multi_label
        self.test_mode = test_mode

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

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        if self.multi_label:
            self._generate_gt_labels()

    def _set_group_flag(self):
        """Set flag according to the mean value (ms) of signal data snrs.

        set as group i, where self.SNRS[i]<=ms<self.SNRS[i+1].
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for idx in range(len(self)):
            ann = self.get_ann_info(idx)
            snrs = ann['gt_snrs']
            mean_snr = np.mean(snrs)
            self.flag[idx] = np.where(np.array(self.SNRS) <= mean_snr)[0][-1]

    def _generate_gt_labels(self):
        self.mix_gt_labels = np.zeros(
            (len(self), len(self.CLASSES)), dtype=np.float32)
        for idx in range(len(self)):
            ann = self.get_ann_info(idx)
            labels = ann['gt_labels']
            for class_index in labels:
                self.mix_gt_labels[idx, class_index] = 1

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        annos = IOLoad(ann_file)
        anno_data = self._label_data_by_snr(annos['data'])
        mods_dict = {'low': 1, 'high': 0}
        snrs_dict = annos['snrs']
        filter_config = annos['filters'][self.filter_config]

        return anno_data, mods_dict, snrs_dict, filter_config

    def _label_data_by_snr(self, anno_data):
        res = []
        for item in anno_data:
            data = dict()
            data['filename'] = item['filename']
            data['ann'] = dict()
            data['ann']['labels'] = []
            data['ann']['snrs'] = item['ann']['snrs']
            for snr in item['ann']['snrs']:
                if snr < -8.0:
                    data['ann']['labels'].append(1)
                else:
                    data['ann']['labels'].append(0)
            res.append(data)

        return res

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
        gt_labels = self.data_infos[idx]['ann']['labels']
        gt_snrs = self.data_infos[idx]['ann']['snrs']
        return self._parse_ann_info(gt_labels, gt_snrs)

    def get_cat_ids(self, idx):
        """Get category ids by idx.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the data of specified idx.
        """

        return self.data_infos[idx]['ann']['labels']

    def _parse_ann_info(self, gt_labels, gt_snrs):
        """Parse labels and snrs annotation

        Args:
            gt_labels (list[int]): label info of an item
            gt_snrs (list[float]): snr info of an item

        Returns:
            dict: A dict containing the following keys: labels <np.ndarray>, snrs <np.ndarray>
        """
        gt_labels = np.array(gt_labels, dtype=np.int64)
        gt_snrs = np.array(gt_snrs, dtype=np.float32)
        ann = dict(gt_labels=gt_labels, gt_snrs=gt_snrs)

        return ann

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
            iq_path = osp.join(
                self.data_root, 'sequence_data', 'iq', file_name)
            iq_data = np.load(iq_path)
            iq_data = iq_data.astype(np.float32)
            iq_data = np.reshape(iq_data, (1, 2, -1))
            data['iqs'] = iq_data

        if self.ap:
            ap_path = osp.join(
                self.data_root, 'sequence_data', 'ap', file_name)
            ap_data = np.load(ap_path)
            ap_data = ap_data.astype(np.float32)
            ap_data = np.reshape(ap_data, (1, 2, -1))
            data['aps'] = ap_data

        if self.co:
            co_path = osp.join(self.data_root, 'constellation_data',
                               'filter_size_{:<.3f}_stride_{:<.3f}'.format(self.filter[0], self.filter[1]), file_name)
            co_data = np.load(co_path)
            co_data = co_data.astype(np.float32)
            co_data = np.expand_dims(co_data, axis=0)
            data['cos'] = co_data

        ann = self.get_ann_info(idx)
        data['gt_labels'] = ann['gt_labels']

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
            iq_path = osp.join(
                self.data_root, 'sequence_data', 'iq', file_name)
            iq_data = np.load(iq_path)
            iq_data = iq_data.astype(np.float32)
            iq_data = np.reshape(iq_data, (1, 2, -1))
            data['iqs'] = iq_data

        if self.ap:
            ap_path = osp.join(
                self.data_root, 'sequence_data', 'ap', file_name)
            ap_data = np.load(ap_path)
            ap_data = ap_data.astype(np.float32)
            ap_data = np.reshape(ap_data, (1, 2, -1))
            data['aps'] = ap_data

        if self.co:
            co_path = osp.join(self.data_root, 'constellation_data',
                               'filter_size_{:<.3f}_stride_{:<.3f}'.format(self.filter[0], self.filter[1]), file_name)
            co_data = np.load(co_path)
            co_data = co_data.astype(np.float32)
            co_data = np.expand_dims(co_data, axis=0)
            data['cos'] = co_data

        return data

    def _evaluate(self, results):
        confusion_matrix = np.zeros(
            (len(self.SNRS), len(self.CLASSES)), dtype=np.float64)

        for idx in range(len(self)):
            ann = self.get_ann_info(idx)
            snrs = ann['gt_snrs']
            labels = ann['gt_labels']
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
            eval_results['snr_{:.3f}_acc'.format(
                snr)] = confusion_matrix[snr_index, 0]
        conf = np.sum(confusion_matrix, axis=0) / len(self.SNRS)
        eval_results['snr_acc'] = conf[0]

        return eval_results

    def _evaluate_mix(self, results):

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
                                                                         self.mix_gt_labels[:, class_ids])

        eval_results = dict()
        for class_ids, class_name in enumerate(self.CLASSES):
            eval_results['{}_thr'.format(
                class_name)] = best_thresholds[class_ids]
            eval_results['{}_acc'.format(
                class_name)] = classification_accuracys[class_ids]

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

        json_results = []
        for idx in range(len(self)):
            data = dict()
            item_info = self.data_infos[idx]
            file_name = item_info['filename']
            data[file_name] = dict()
            data[file_name]['label'] = results[idx]
            json_results.append(data)

        IODump(json_results, osp.join(out_dir, 'res.json'))

    def evaluate(self, results, logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        """
        results = [result.reshape(1, len(self.CLASSES)) for result in results]
        results = np.concatenate(results, axis=0)
        results = np.reshape(results, (len(self), -1))

        print(np.isnan(results).any())

        if not self.multi_label:
            eval_results = self._evaluate(results)
        else:
            eval_results = self._evaluate_mix(results)

        return eval_results
