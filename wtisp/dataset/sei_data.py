import copy
import os
import os.path as osp
import pickle

import numpy as np
import tqdm
from torch.utils.data import Dataset

from .builder import DATASETS
from ..common.fileio import dump as IODump
from ..common.fileio import load as IOLoad


@DATASETS.register_module()
class WTISEIDataset(Dataset):
    """Base dataset for modulation classification.

    The annotation format is shown as follows. The `ann` field is optional for
    testing.

        .. code-block:: none

            [
                data:[
                    {
                        'filename': '00228.npy',
                        'ann':{
                            'mod': int,
                            'dev': int,
                        }

                    }
                    ...GG
                ]

                mods:{
                    'bpsk': 0,
                    ...
                }

                devs:{
                    -10: 0,
                    ...
                }
            ]

    Args:
        ann_file (str): Annotation file path.

    """
    CLASSES = None
    MODS = None

    def __init__(self, ann_file, is_iq=False, is_ap=False, is_co=True, dif_step=0, item_length=160000,
                 truncation_length=10000, test_mode=False, modulation_list=None, device_list=None, data_root=None,
                 filter_size=None, filter_stride=None, is_dual=False, return_dual_label=False):

        self.ann_file = ann_file
        self.is_iq = is_iq
        self.is_ap = is_ap
        self.is_co = is_co
        self.dif_step = dif_step
        self.item_length = item_length
        self.data_root = data_root
        self.truncation_length = truncation_length
        self.test_mode = test_mode
        self.modulation_list = modulation_list
        self.device_list = device_list
        self.filter_size = filter_size
        self.filter_stride = filter_stride
        self.num_slices_per_item = int(item_length // truncation_length)
        self.width_range = [-1, 1]
        self.height_range = [-1, 1]
        # TODO: the 'is_dual' should be more elegant
        self.is_dual = is_dual
        if return_dual_label:
            self.is_dual = True
            self.return_dual_label = True
        else:
            self.return_dual_label = False

        if self.dif_step is 0:
            self.is_dif = False
        else:
            self.is_dif = True

        if self.modulation_list is None:
            self.modulation_list = ['bpsk']
        if self.device_list is None:
            self.device_list = ['dev1', 'dev2', 'dev3', 'dev4', 'dev5']

        if self.filter_size is None:
            self.filter_size = 0.015625
        if self.filter_stride is None:
            self.filter_stride = 0.015625

        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)

        # load annotations
        self.data_infos, self.mod_dict, self.dev_dict, self.index_mod_dict, self.index_dev_dict = self.load_annotations(
            self.ann_file)

        self.MODS = list(self.mod_dict.keys())
        self.CLASSES = list(self.dev_dict.keys())

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        pkl_dir = os.path.join(self.data_root, 'cache_pkl')
        if not os.path.isdir(pkl_dir):
            os.makedirs(pkl_dir)

        if self.test_mode:
            cache_data_path = os.path.join(pkl_dir, '_'.join(self.modulation_list) + '_test.pkl')
        else:
            cache_data_path = os.path.join(pkl_dir, '_'.join(self.modulation_list) + '_train_and_val.pkl')

        if os.path.isfile(cache_data_path):
            self.cache_data = pickle.load(open(cache_data_path, 'rb'))
        else:
            self.cache_data = self.cache_all()
            pickle.dump(self.cache_data, open(cache_data_path, 'wb'))

        if self.is_dual:
            self.dev_samples = dict()
            for item_index, ann in enumerate(self.data_infos):
                if ann['ann']['dev'] in self.dev_samples:
                    self.dev_samples[ann['ann']['dev']].append(dict(filename=ann['filename'], item_index=item_index))
                else:
                    self.dev_samples[ann['ann']['dev']] = [dict(filename=ann['filename'], item_index=item_index)]
        else:
            self.dev_samples = None

    def _set_group_flag(self):
        """Set flag according to the mean value (ms) of signal data snrs.

        set as group i, where self.SNRS[i]<=ms<self.SNRS[i+1].
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos) * self.num_slices_per_item

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        annos = IOLoad(ann_file)
        anno_data = annos['data']
        mods_dict = annos['mods']
        devs_dict = annos['devs']

        rmods_dict = {v: k for k, v in mods_dict.items()}
        rdevs_dict = {v: k for k, v in devs_dict.items()}

        # filter data
        save_anno_data = []
        for item in anno_data:
            if (rdevs_dict[item['ann']['dev']] in self.device_list) and (
                    rmods_dict[item['ann']['mod']] in self.modulation_list):
                save_anno_data.append(copy.deepcopy(item))

        save_mods_dict = {v: i for i, v in enumerate(self.modulation_list)}
        save_devs_dict = {v: i for i, v in enumerate(self.device_list)}
        return save_anno_data, save_mods_dict, save_devs_dict, rmods_dict, rdevs_dict

    def filter_item(self, idx):
        item = self.cache_data[idx]
        # https://stackoverflow.com/questions/11277432/how-can-i-remove-a-key-from-a-python-dictionary
        if not self.is_iq:
            item.pop('iqs', None)
        if not self.is_ap:
            item.pop('aps', None)
        if not self.is_co:
            item.pop('cos', None)

        return item

    def __getitem__(self, idx):

        item1 = self.filter_item(idx)
        if self.test_mode:
            # In test mode, the contrastive loss is useless,
            # we only concentrate on the outputs from the
            # classifier head so the self.is_dual is not working
            return item1

        if self.is_dual:
            p = np.array([0.5, 0.5])
            from_same_dev = np.random.choice([0, 1], p=p.ravel())
            item_index = idx // self.num_slices_per_item
            ann = self.data_infos[item_index]
            dev_label = ann['ann']['dev']
            if from_same_dev:
                tmp_sample = dict(filename=ann['filename'], item_index=item_index)
                sample_pool = copy.deepcopy(self.dev_samples[dev_label])
                sample_pool.remove(tmp_sample)
                sample = np.random.choice(sample_pool)
                sample_index = sample['item_index']
                splice_index = np.random.choice([i for i in range(self.num_slices_per_item)])
                sample_idx = sample_index * self.num_slices_per_item + splice_index
                item2 = self.filter_item(sample_idx)
            else:
                sample_pool = []
                for dev_label_tmp in self.dev_samples:
                    if dev_label_tmp is not dev_label:
                        sample_pool.extend(copy.deepcopy(self.dev_samples[dev_label_tmp]))
                sample = np.random.choice(sample_pool)
                sample_index = sample['item_index']
                splice_index = np.random.choice([i for i in range(self.num_slices_per_item)])
                sample_idx = sample_index * self.num_slices_per_item + splice_index
                item2 = self.filter_item(sample_idx)
            if self.return_dual_label:
                return [item1, item2, dict(from_same_dev=np.array(from_same_dev))]
            else:
                return [item1, item2]
        else:
            return item1

    def get_ann_info(self, idx):
        """Get annotation by idx.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified idx.
        """
        ann_info_key = []
        ann_info_val = []

        mod_label = self.data_infos[idx]['ann']['mod']
        ann_info_key.append('mod_label')
        ann_info_val.append(self.mod_dict[self.index_mod_dict[mod_label]])

        dev_label = self.data_infos[idx]['ann']['dev']
        ann_info_key.append('dev_label')
        ann_info_val.append(self.dev_dict[self.index_dev_dict[dev_label]])
        return self._parse_ann_info(ann_info_key, ann_info_val)

    def get_cat_ids(self, idx):
        """Get category ids by idx.

        Args:
            idx (int): Index of data.

        Returns:
            int: The categories in the data of specified idx.
        """

        return self.data_infos[idx]['ann']['devs']

    @staticmethod
    def _parse_ann_info(ann_info_key, ann_info_val):
        """Parse mod and dev annotation

        Args:
            ann_info_key (list[str]): ann name of an item
            ann_info_val (list[int]): ann val of an item

        Returns:
            dict: A dict
        """
        ann = dict()
        for i, key_name in enumerate(ann_info_key):
            ann[key_name] = np.array(ann_info_val[i], dtype=np.int64)

        return ann

    def convert_iq_to_co(self, data):

        matrix_width = int((self.width_range[1] - self.width_range[0] - self.filter_size) / self.filter_stride + 1)
        matrix_height = int((self.height_range[1] - self.height_range[0] - self.filter_size) / self.filter_stride + 1)

        constellation = np.zeros((matrix_height, matrix_width), dtype=np.float32)

        def axis_is(query_axis_x, query_axis_y):
            axis_x = query_axis_x // self.filter_stride
            axis_y = query_axis_y // self.filter_stride
            if axis_x * self.filter_stride + self.filter_size < query_axis_x:
                position = [None, None]
            elif axis_y * self.filter_stride + self.filter_size < query_axis_y:
                position = [None, None]
            else:
                position = [int(axis_x), int(axis_y)]
            return position

        pos_list = map(axis_is, list(data[0, :]), list(data[1, :]))
        for pos in pos_list:
            if pos[0] is not None:
                constellation[pos[0], pos[1]] += 1

        return constellation

    def convert_complex_to_iq_ap(self, data):
        iq_data = np.concatenate((np.array(data.real, ndmin=2).T, np.array(data.imag, ndmin=2).T), axis=0)

        amplitude = np.sqrt(np.sum(np.power(iq_data, 2), axis=0))
        phase = np.arctan(iq_data[0, :] / (iq_data[1, :] + np.finfo(np.float64).eps))
        ap_data = np.vstack((amplitude, phase))

        return iq_data, ap_data

    def cache_item(self, item_index):
        item_info = self.data_infos[item_index]
        item_data = np.load(os.path.join(self.data_root, self.index_mod_dict[item_info['ann']['mod']],
                                         self.index_dev_dict[item_info['ann']['dev']], item_info['filename']))

        cache_list = []
        for i in range(self.num_slices_per_item):
            sub_item_data = copy.deepcopy(item_data[i * self.truncation_length:(i + 1) * self.truncation_length])
            iq_data, ap_data = self.convert_complex_to_iq_ap(sub_item_data)

            if self.is_dif:
                right_index = self.truncation_length - self.dif_step
                left_index = self.dif_step + 1
                sub_item_data = sub_item_data[:right_index, 0] * sub_item_data[left_index:, 0].conjugate()

            sub_item_data = np.concatenate(
                (np.array(sub_item_data.real, ndmin=2).T, np.array(sub_item_data.imag, ndmin=2).T), axis=0)
            sub_item_data[0, :] = sub_item_data[0, :] / np.max(np.abs(sub_item_data[0, :]))
            sub_item_data[1, :] = sub_item_data[1, :] / np.max(np.abs(sub_item_data[1, :]))

            co_data = self.convert_iq_to_co(sub_item_data)
            co_data = np.expand_dims(co_data, axis=0)
            if self.test_mode:
                cache_list.append(dict(iqs=iq_data, aps=ap_data, cos=co_data))
            else:
                ann = self.get_ann_info(item_index)
                cache_list.append(dict(iqs=iq_data, aps=ap_data, cos=co_data, dev_labels=ann['dev_label']))

        return cache_list

    def cache_all(self):
        cache_data = []
        for item_index in tqdm.tqdm(range(len(self.data_infos))):
            data = self.cache_item(item_index)
            cache_data.extend(data)
        return cache_data

    def _evaluate_dev(self, results, prefix=''):
        confusion_matrix = np.zeros((len(self.MODS), len(self.CLASSES), len(self.CLASSES)), dtype=np.float64)

        for idx in range(len(self)):
            item_index = idx // self.num_slices_per_item
            ann = self.get_ann_info(item_index)
            dev_label = ann['dev_label']
            mod_label = ann['mod_label']
            predict_dev_index = int(np.argmax(results[idx, :]))
            confusion_matrix[mod_label, dev_label, predict_dev_index] += 1

        confusion_matrix = confusion_matrix / np.expand_dims(np.sum(confusion_matrix, axis=2), axis=2)

        eval_results = dict()
        for mod in self.MODS:
            conf = confusion_matrix[self.mod_dict[mod], :, :]
            cor = np.sum(np.diag(conf))
            ncor = np.sum(conf) - cor
            eval_results[prefix + 'mod_{}'.format(mod)] = 1.0 * cor / (cor + ncor)
        conf = np.sum(confusion_matrix, axis=0) / len(self.MODS)
        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor
        eval_results[prefix + 'mod_mean_all'] = 1.0 * cor / (cor + ncor)
        return eval_results

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

        eval_results = self._evaluate_dev(results, prefix='common/')

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

        def format_out_single(in_results, in_save_path):
            in_results = [result.reshape(1, len(self.CLASSES))
                          for result in in_results]
            in_results = np.concatenate(in_results, axis=0)
            in_results = np.reshape(in_results, (len(self), -1))
            np.save(in_save_path, in_results)

        save_path = os.path.join(out_dir, 'pre.npy')
        format_out_single(results, save_path)

        json_results = dict(CLASSES=self.CLASSES, MODS=self.MODS,
                            mods_dict=self.mod_dict, devs_dict=self.dev_dict, ANN=[])
        for idx in range(len(self)):
            item_index = idx // self.num_slices_per_item
            ann = self.get_ann_info(item_index)
            json_results['ANN'].append(ann)

        IODump(json_results, osp.join(out_dir, 'ann.json'))
