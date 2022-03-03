import os
import os.path as osp
import pickle

import numpy as np
from scipy.special import softmax
from torch.utils.data import Dataset
from tqdm import tqdm

from .builder import DATASETS
from .merge import get_merge_weight_by_optimization
from ..common.fileio import dump as IODump
from ..common.fileio import load as IOLoad


def generate_targets(ann_file):
    annos = IOLoad(ann_file)
    anno_data = annos['data']
    mods_dict = annos['mods']
    targets = np.zeros((len(anno_data), len(mods_dict.keys())), dtype=np.float64)
    for item_index, item in enumerate(anno_data):
        targets[item_index, mods_dict[item['ann']['labels']]] = 1

    return targets


@DATASETS.register_module()
class WTIMCOnlineDataset(Dataset):
    CLASSES = None

    def __init__(self, ann_file, data_root=None, test_mode=False, channel_mode=False, use_cache=False, merge_res=None):
        self.ann_file = ann_file
        self.data_root = data_root
        self.test_mode = test_mode
        self.channel_mode = channel_mode
        self.use_cache = use_cache
        self.merge_res = merge_res

        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)

        # load annotations
        self.data_infos, self.mods_dict = self.load_annotations(self.ann_file)
        self.index_class_dict = {index: mod for mod,
                                                index in self.mods_dict.items()}
        self.CLASSES = [''] * len(self.index_class_dict)

        for class_index in self.index_class_dict.keys():
            self.CLASSES[class_index] = self.index_class_dict[class_index]

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        # cache data in init
        if self.use_cache:
            pkl_dir = os.path.join(self.data_root, 'cache_pkl')
            if not os.path.isdir(pkl_dir):
                os.makedirs(pkl_dir)

            if self.test_mode:
                cache_data_path = os.path.join(pkl_dir, 'test')
            else:
                if 'train_and_val' in ann_file:
                    cache_data_path = os.path.join(pkl_dir, 'train_and_val')
                elif 'train' in ann_file:
                    cache_data_path = os.path.join(pkl_dir, 'train')
                elif 'val' in ann_file:
                    cache_data_path = os.path.join(pkl_dir, 'val')
                else:
                    raise ValueError(
                        'Please set correct data mode ["train_and_val", "train", "val", "test"] in {}'.format(ann_file))

            self.cache_data = dict()
            self.load_cache_data_in_init('iq', cache_data_path + '-iq.pkl')

        # the targets are for knowledge distillation
        self.targets = generate_targets(self.ann_file)

    def load_cache_data_in_init(self, data_type, cache_data_path=None):
        if os.path.isfile(cache_data_path):
            self.cache_data[data_type] = pickle.load(open(cache_data_path, 'rb'))
        else:
            data_list = []
            for idx in tqdm(range(len(self))):
                item_info = self.data_infos[idx]
                file_name = item_info['filename']
                data = self.load_data_from_file(data_type, file_name)
                data = data.astype(np.float32)
                data_list.append(data)
            pickle.dump(data_list, open(cache_data_path, 'wb'))
            self.cache_data[data_type] = data_list

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
        return len(self.data_infos)

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        annos = IOLoad(ann_file)
        anno_data = annos['data']
        mods_dict = annos['mods']

        return anno_data, mods_dict

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

        return self._parse_ann_info(ann_info_keys, ann_info_vals)

    def get_cat_ids(self, idx):
        """Get category ids by idx.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the data of specified idx.
        """

        return self.data_infos[idx]['ann']['labels']

    def _parse_ann_info(self, ann_info_keys, ann_info_vals):
        """Parse labels and snrs annotation

        Args:
            ann_info_keys (list[str]): ann name of an item
            ann_info_vals (list[float]): ann val of an item

        Returns:
            dict: A dict
        """
        ann = dict()
        for i, key_name in enumerate(ann_info_keys):
            ann[key_name] = np.array(self.mods_dict[ann_info_vals[i]], dtype=np.int64)

        return ann

    def load_data_from_cache(self, data_type, idx=-1):
        return self.cache_data[data_type][idx]

    def load_data_from_file(self, data_type, file_name):
        file_path = osp.join(
            self.data_root, 'sequence_data', data_type, file_name)
        seq_data = np.load(file_path)
        return seq_data

    def load_data(self, data_type, file_name, idx=-1):

        if self.use_cache:
            data = self.load_data_from_cache(data_type, idx)
        else:
            data = self.load_data_from_file(data_type, file_name)

        data = data.astype(np.float32)
        num_path = 2
        if self.channel_mode:
            data = np.reshape(data, (num_path, 1, -1))
        else:
            data = np.reshape(data, (1, num_path, -1))

        return data

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
        iq_data = self.load_data('iq', file_name, idx)
        data['iqs'] = iq_data

        ann = self.get_ann_info(idx)
        data['mod_labels'] = ann['mod_labels']

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

        iq_data = self.load_data('iq', file_name, idx)
        data['iqs'] = iq_data

        return data

    def _evaluate_mod(self, results, prefix=''):
        confusion_matrix = np.zeros((len(self.CLASSES), len(self.CLASSES)), dtype=np.float64)

        for idx in range(len(self)):
            ann = self.get_ann_info(idx)
            label = ann['mod_labels']
            predict_class_index = int(np.argmax(results[idx, :]))
            confusion_matrix[label, predict_class_index] += 1

        eval_results = dict()
        cor = np.sum(np.diag(confusion_matrix))
        ncor = np.sum(confusion_matrix) - cor
        eval_results[prefix + 'snr_mean_all'] = 1.0 * cor / (cor + ncor)
        return eval_results

    def _reshape_result(self, results, num_cols):
        results = [result.reshape(1, num_cols) for result in results]
        results = np.concatenate(results, axis=0)
        results = np.reshape(results, (len(self), -1))
        return results

    def process_single_head(self, results, prefix=''):
        results = self._reshape_result(results, len(self.CLASSES))
        eval_results = self._evaluate_mod(results, prefix=prefix)

        return eval_results

    def _process_merge(self, results_dict, prefix=''):
        pre_matrix = []
        for key_str in results_dict:
            pre_data = results_dict[key_str]
            pre_data = self._reshape_result(pre_data, len(self.CLASSES))
            pre_data = softmax(pre_data, axis=1)
            pre_data = pre_data[None, :, :]
            pre_matrix.append(pre_data)

        pre_matrix = np.concatenate(pre_matrix, axis=0)
        pre_max_index = np.argmax(pre_matrix, axis=2)
        pre_max_index = np.sum(pre_max_index, axis=0)
        gt_max_index = np.argmax(self.targets, axis=1) * len(pre_matrix)
        no_zero_index = np.nonzero((pre_max_index - gt_max_index))[0]

        bad_pre_matrix = pre_matrix[:, no_zero_index[:], :]
        targets = self.targets[no_zero_index[:], :]

        #
        w = get_merge_weight_by_optimization(bad_pre_matrix, targets)
        merge_matrix = np.dot(w.T, np.reshape(pre_matrix, (len(results_dict), -1)))
        merge_matrix = np.reshape(merge_matrix, (-1, len(self.CLASSES)))
        eval_results = self._evaluate_mod(merge_matrix, prefix='final/')

        return eval_results, merge_matrix

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
                if 'fea' in key_str:
                    continue
                sub_eval_results = self.process_single_head(
                    format_results[key_str], prefix=key_str + '/')
                eval_results.update(sub_eval_results)

            if self.merge_res and len(format_results) > 1:
                merge_eval_results, _ = self._process_merge(format_results)
                eval_results.update(merge_eval_results)
        else:
            eval_results = self.process_single_head(results, prefix='final/')

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

        def format_out_single(res, file_path):
            res = [result.reshape(1, -1) for result in res]
            res = np.concatenate(res, axis=0)
            res = np.reshape(res, (len(self), -1))
            np.save(file_path, res)

        if isinstance(results[0], dict):
            format_results = {key_str: [] for key_str in results[0]}
            for item in results:
                for key_str in item:
                    format_results[key_str].append(item[key_str])

            for key_str in format_results.keys():
                sub_results = format_results[key_str]
                save_path = os.path.join(out_dir, key_str + '.npy')
                format_out_single(sub_results, save_path)

            if self.merge_res and len(format_results) > 1:
                save_path = os.path.join(out_dir, 'pre.npy')
                _, merge_matrix = self._process_merge(format_results)
                format_out_single(merge_matrix, save_path)
        else:
            save_path = os.path.join(out_dir, 'pre.npy')
            format_out_single(results, save_path)

        json_results = dict(CLASSES=self.CLASSES, mods_dict=self.mods_dict, ANN=[])
        for idx in range(len(self)):
            ann = self.get_ann_info(idx)
            json_results['ANN'].append(ann)

        IODump(json_results, osp.join(out_dir, 'ann.json'))
