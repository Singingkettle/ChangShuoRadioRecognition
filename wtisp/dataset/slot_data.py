import os
import os.path as osp
import pickle
import zlib
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from .builder import DATASETS
from ..common.fileio import dump as IODump
from ..common.fileio import load as IOLoad


@DATASETS.register_module()
class SlotDataset(Dataset):
    """Base dataset for modulation classification.

    The annotation format is shown as follows. The `ann` field is optional for
    testing.

        .. code-block:: none

            [
                data:[
                    {
                        'filename': '0-000000162059.npy',
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
        filter_config (int, optional): the idx to decide which kind of constellation data
        should be used. Default is 0.
        data_root (str, optional): Data root for ``ann_file``, ``In-phase/Quadrature data``,
        ``Amplitude/Phase data``, ``constellation data``.
        confusion_plot (bool, optional): If plot the confusion matrix
        multi_label (bool, optional): If set true, the input signal is the multi_label of different signals with
        different modulation and snr.
        test_mode (bool, optional): If set True, annotation will not be loaded.
        snr_threshold (float, optional): When the use_snr_lables is True, the item in dataset will be added extra info to indicate its snr property
        item_weights (list, optional): If the item is low snr, then the high head scale the gradient by a small value
    """
    CLASSES = None
    SNRS = None

    def __init__(self,
                 ann_file,
                 data_root=None,
                 confusion_plot=False,
                 test_mode=False,
                 snr_threshold=-6,
                 use_snr_label=False,
                 item_weights=[0.1, 0.9],
                 is_seq=False,
                 use_ap=False,
                 channel_mode=False):

        self.ann_file = ann_file
        self.data_root = data_root
        self.confusion_plot = confusion_plot
        self.test_mode = test_mode
        self.snr_threshold = snr_threshold
        self.use_snr_label = use_snr_label
        self.item_weights = item_weights
        self.is_seq = is_seq
        self.use_ap = use_ap
        self.channel_mode = channel_mode

        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)

        # load annotations
        self.data_infos, self.mods_dict, self.snrs_dict = self.load_annotations(
            self.ann_file)
        self.index_class_dict = {
            index: mod
            for mod, index in self.mods_dict.items()
        }
        self.index_snr_dict = {
            index: float(snr)
            for snr, index in self.snrs_dict.items()
        }

        self.CLASSES = [''] * len(self.index_class_dict)
        self.SNRS = [0.0] * len(self.index_snr_dict)

        for class_index in self.index_class_dict.keys():
            self.CLASSES[class_index] = self.index_class_dict[class_index]
        for snr_index in self.index_snr_dict.keys():
            self.SNRS[snr_index] = self.index_snr_dict[snr_index]

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        # load data
        if self.test_mode:
            cache_data_path = os.path.join(self.data_root, 'test.pkl')
        else:
            cache_data_path = os.path.join(self.data_root, 'train.pkl')
        if os.path.isfile(cache_data_path):
            self.cache_data = pickle.load(open(cache_data_path, 'rb'))
        else:
            self.cache_data = []
            for idx in tqdm(range(len(self))):
                item_info = self.data_infos[idx]

                file_name = item_info['filename']
                data = {}
                item_path = osp.join(self.data_root, 'data', file_name)
                item = np.load(item_path)
                if self.is_seq:
                    item = np.reshape(item, (2, -1))
                    item = item[:, 10000:10128]
                    iq_item = item.astype(np.float32)
                    ap_item = item[0, :] + 1j * item[1, :]
                    amplitude = np.abs(ap_item)
                    phase = np.angle(ap_item)
                    ap_item = np.vstack((amplitude, phase))
                    data['iqs'] = iq_item
                    data['aps'] = ap_item
                else:
                    item = item.astype(np.int16)
                    item = zlib.compress(item.tostring())
                    data['cos'] = item
                self.cache_data.append(data)
            pickle.dump(self.cache_data, open(cache_data_path, 'wb'))
            if self.is_seq and (not self.use_ap):
                for item in self.cache_data:
                    del item['aps']

    def _set_group_flag(self):
        """Set flag according to the mean value (ms) of signal data snrs.

        set as group i, where self.SNRS[i]<=ms<self.SNRS[i+1].
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def _generate_mod_labels(self):
        self.mix_mod_labels = np.zeros((len(self), len(self.CLASSES)),
                                       dtype=np.float32)
        for idx in range(len(self)):
            ann = self.get_ann_info(idx)
            labels = ann['mod_labels']
            for class_index in labels:
                self.mix_mod_labels[idx, class_index] = 1

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        annos = IOLoad(ann_file)
        anno_data = annos['data']
        mods_dict = annos['mods']
        snrs_dict = annos['snrs']
        if self.use_snr_label:
            tmp_anno_data = []
            for item in anno_data:
                if item['ann']['snrs'][0] >= self.snr_threshold:
                    item['ann']['snr_labels'] = [0]
                    item['ann']['low_weight'] = [self.item_weights[0]]
                    item['ann']['high_weight'] = [self.item_weights[1]]
                else:
                    item['ann']['snr_labels'] = [1]
                    item['ann']['low_weight'] = [self.item_weights[1]]
                    item['ann']['high_weight'] = [self.item_weights[0]]
                tmp_anno_data.append(item)
            anno_data = tmp_anno_data

        return anno_data, mods_dict, snrs_dict

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
        mod_labels = self.data_infos[idx]['ann']['labels']
        snrs = self.data_infos[idx]['ann']['snrs']
        if self.use_snr_label:
            snr_labels = self.data_infos[idx]['ann']['snr_labels']
            low_weight = self.data_infos[idx]['ann']['low_weight']
            high_weight = self.data_infos[idx]['ann']['high_weight']
            return self.__parse_ann_info(mod_labels, snrs, snr_labels,
                                         low_weight, high_weight)
        else:
            return self._parse_ann_info(mod_labels, snrs)

    def get_cat_ids(self, idx):
        """Get category ids by idx.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the data of specified idx.
        """

        return self.data_infos[idx]['ann']['labels']

    def _parse_ann_info(self, mod_labels, snrs):
        """Parse labels and snrs annotation

        Args:
            mod_labels (list[int]): label info of an item
            snrs (list[float]): snr info of an item

        Returns:
            dict: A dict containing the following keys: labels <np.ndarray>, snrs <np.ndarray>
        """
        mod_labels = np.array(mod_labels, dtype=np.int64)
        snrs = np.array(snrs, dtype=np.float32)
        ann = dict(mod_labels=mod_labels, snrs=snrs)

        return ann

    def __parse_ann_info(self, mod_labels, snrs, snr_labels, low_weight,
                         high_weight):
        """Parse labels and snrs annotation

        Args:
            mod_labels (list[int]): label info of an item
            snrs (list[float]): snr info of an item
            snr_labels (list[int]): snr label info of an item
            low_weight (list[float]): If the item is a low snr data, then the low weight is large than high weight
            high_weight (list[float]): If the item is a high snr data, then the high weight is large than low weight
        Returns:
            dict: A dict containing the following keys: labels <np.ndarray>, snrs <np.ndarray>
        """
        mod_labels = np.array(mod_labels, dtype=np.int64)
        snrs = np.array(snrs, dtype=np.float32)
        snr_labels = np.array(snr_labels, dtype=np.int64)
        low_weight = np.array(low_weight, dtype=np.float32)
        high_weight = np.array(high_weight, dtype=np.float32)

        ann = dict(mod_labels=mod_labels,
                   snrs=snrs,
                   snr_labels=snr_labels,
                   low_weight=low_weight,
                   high_weight=high_weight)

        return ann

    def load_compress_data(self, idx, item_info, data_type):
        data = self.cache_data[idx][data_type]
        item = deepcopy(data)
        if self.is_seq:
            item = item.astype(np.float32)
            if self.channel_mode:
                item = np.reshape(item, (2, 1, -1))
            else:
                item = np.reshape(item, (1, 2, -1))
        else:
            try:
                item = np.frombuffer(zlib.decompress(item), dtype=np.int16).reshape([8, 128, 128])
                item = item.astype(np.float32)
            except:
                file_name = item_info['filename']
                item_path = osp.join(self.data_root, 'data', file_name)
                item = np.load(item_path)
                item = item.astype(np.float32)
        return item

    def prepare_train_data(self, idx):
        """Get training data.

        Args.
         idx (int): Index of data

        Returns:
            dict: Training data and annotation
        """

        item_info = self.data_infos[idx]

        data = {}
        if self.is_seq:
            data['iqs'] = self.load_compress_data(idx, item_info, 'iqs')
            if self.use_ap:
                data['aps'] = self.load_compress_data(idx, item_info, 'aps')
        else:
            data['cos'] = self.load_compress_data(idx, item_info, 'cos')
        ann = self.get_ann_info(idx)
        data['mod_labels'] = ann['mod_labels']
        if self.use_snr_label:
            data['snr_labels'] = ann['snr_labels']
            data['low_weight'] = ann['low_weight']
            data['high_weight'] = ann['high_weight']

        return data

    def prepare_test_data(self, idx):
        """Get testing data.

        Args.
         idx (int): Index of data

        Returns:
            dict: Testing data and annotation
        """
        item_info = self.data_infos[idx]
        data = {}
        if self.is_seq:
            data['iqs'] = self.load_compress_data(idx, item_info, 'iqs')
            if self.use_ap:
                data['aps'] = self.load_compress_data(idx, item_info, 'aps')
        else:
            data['cos'] = self.load_compress_data(idx, item_info, 'cos')

        return data

    def _plot_confusion_matrix(self,
                               cm,
                               title='Confusion matrix',
                               cmap=plt.cm.Blues,
                               labels=[]):
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

    def _evaluate(self, results, prefix=''):
        confusion_matrix = np.zeros(
            (len(self.SNRS), len(self.CLASSES), len(self.CLASSES)),
            dtype=np.float64)

        for idx in range(len(self)):
            ann = self.get_ann_info(idx)
            snrs = ann['snrs']
            labels = ann['mod_labels']
            if snrs.size == 1 and labels.size == 1:
                predict_class_index = int(np.argmax(results[idx, :]))
                confusion_matrix[self.snrs_dict['{:.3f}'.format(snrs[0])],
                                 labels[0], predict_class_index] += 1
            else:
                raise ValueError(
                    'Please check your dataset, the size of snrs and labels are both 1 for any item. '
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
                eval_results[prefix + 'snr_{:.3f}_conf_figure'.format(
                    snr)] = self._plot_confusion_matrix(
                    conf,
                    title='Confusion matrix (SNR={:.3f})'.format(snr),
                    labels=self.CLASSES)
        conf = np.sum(confusion_matrix, axis=0) / len(self.SNRS)
        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor
        eval_results[prefix + 'snr_mean_all'] = 1.0 * cor / (cor + ncor)
        if self.confusion_plot:
            eval_results[
                prefix +
                'snr_mean_all_conf_figure'] = self._plot_confusion_matrix(
                conf,
                title='Mean of all snr Confusion matrix',
                labels=self.CLASSES)
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

        results = [result.reshape(1, len(self.CLASSES)) for result in results]
        results = np.concatenate(results, axis=0)
        results = np.reshape(results, (len(self), -1))
        save_path = os.path.join(out_dir, 'pre.npy')
        np.save(save_path, results)

        json_results = dict(CLASSES=self.CLASSES,
                            SNRS=self.SNRS,
                            mods_dict=self.mods_dict,
                            snrs_dict=self.snrs_dict,
                            ANN=[])
        for idx in range(len(self)):
            ann = self.get_ann_info(idx)
            json_results['ANN'].append(ann)

        IODump(json_results, osp.join(out_dir, 'ann.json'))

    def _evaluate_snr(self, results, prefix):
        confusion_matrix = np.zeros((len(self.SNRS), len(self.item_weights)),
                                    dtype=np.float64)

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
                raise ValueError(
                    'Please check your dataset, the size of snrs and labels are both 1 for any item. '
                    'However, the current item with the idx {:d} has the snrs size {:d} and the '
                    'labels size {:d}'.format(idx, snrs.size, labels.size))

        confusion_matrix = confusion_matrix / \
                           np.expand_dims(np.sum(confusion_matrix, axis=1), axis=1)

        eval_results = dict()
        for snr_index, snr in enumerate(self.SNRS):
            eval_results[prefix +
                         'snr_{:.3f}_acc'.format(snr)] = confusion_matrix[
                snr_index, 0]
        conf = np.sum(confusion_matrix, axis=0) / len(self.SNRS)
        eval_results[prefix + 'snr_acc'] = conf[0]

        return eval_results

    def process_single_head(self, results, prefix=''):

        if 'snr' in prefix:
            results = [
                result.reshape(1, len(self.item_weights)) for result in results
            ]
            results = np.concatenate(results, axis=0)
            results = np.reshape(results, (len(self), -1))
            eval_results = self._evaluate_snr(results, prefix=prefix)
            return eval_results
        else:
            results = [
                result.reshape(1, len(self.CLASSES)) for result in results
            ]
            results = np.concatenate(results, axis=0)
            results = np.reshape(results, (len(self), -1))

            eval_results = self._evaluate(results, prefix=prefix)

            return eval_results

    def evaluate(self, results, logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        """
        if self.use_snr_label:
            format_results = dict()
            format_results = {key_str: [] for key_str in results[0].keys()}
            for item in results:
                for key_str in item.keys():
                    format_results[key_str].append(item[key_str])

            eval_results = dict()
            for key_str in format_results.keys():
                sub_eval_results = self.process_single_head(
                    format_results[key_str], prefix=key_str + '/')
                eval_results.update(sub_eval_results)
        else:
            eval_results = self.process_single_head(results, prefix='common/')

        return eval_results
