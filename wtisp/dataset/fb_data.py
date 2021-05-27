import os
import os.path as osp

import numpy as np

from .builder import DATASETS
from ..common.fileio import dump as IODump
from ..common.fileio import load as IOLoad


@DATASETS.register_module()
class FBDataset(object):
    CLASSES = None
    SNRS = None

    def __init__(self, ann_file, data_root=None):
        self.ann_file = ann_file
        self.data_root = data_root
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)

        # load annotations
        self.data, self.label, self.data_infos, self.mods_dict, self.snrs_dict = self.load_data(self.ann_file)
        self.index_class_dict = {index: mod for mod, index in self.mods_dict.items()}
        self.index_snr_dict = {index: float(snr) for snr, index in self.snrs_dict.items()}

        self.CLASSES = [''] * len(self.index_class_dict)
        self.SNRS = [0.0] * len(self.index_snr_dict)

        for class_index in self.index_class_dict.keys():
            self.CLASSES[class_index] = self.index_class_dict[class_index]
        for snr_index in self.index_snr_dict.keys():
            self.SNRS[snr_index] = self.index_snr_dict[snr_index]

    def __len__(self):
        """Total number of samples of data."""
        return self.data.shape[0]

    @staticmethod
    def generate_cumulants(iq):
        iq = iq[0, :] + 1j * iq[1, :]
        iq_length = iq.shape[0]
        M20 = np.sum(np.power(iq, 2)) / iq_length
        M21 = np.sum(np.power(np.abs(iq), 2)) / iq_length
        M22 = np.sum(np.power(np.conj(iq), 2)) / iq_length
        M40 = np.sum(np.power(iq, 4)) / iq_length
        M41 = np.sum(np.power(iq, 2) * np.power(np.abs(iq), 2)) / iq_length
        M42 = np.sum(np.power(np.abs(iq), 4)) / iq_length
        M43 = np.sum(np.power(np.conj(iq), 2) * np.power(np.abs(iq), 2)) / iq_length
        M60 = np.sum(np.power(iq, 6)) / iq_length
        M61 = np.sum(np.power(iq, 4) * np.power(np.abs(iq), 2)) / iq_length
        M62 = np.sum(np.power(iq, 2) * np.power(np.abs(iq), 4)) / iq_length
        M63 = np.sum(np.power(np.abs(iq), 6)) / iq_length

        C20 = M20
        C21 = M21
        C40 = M40 - 3 * np.power(M20, 2)
        C41 = M41 - 3 * M20 * M21
        C42 = M42 - np.power(np.abs(M20), 2) - 2 * np.power(M21, 2)
        C60 = M60 - 15 * M20 * M40 + 3 * np.power(M20, 3)
        C61 = M61 - 5 * M21 * M40 - 10 * M20 * M41 + 30 * np.power(M20, 2) * M21
        C62 = M62 - 6 * M20 * M42 - 8 * M21 * M41 - M22 * M40 + 6 * np.power(M20, 2) * M22 + 24 * np.power(M21, 2) * M20
        C63 = M63 - 9 * M21 * M42 + 12 * np.power(M21, 3) - 3 * M20 * M43 - 3 * M22 * M41 + 18 * M20 * M21 * M22

        C20_norm = C20 / np.power(C21, 2)
        C21_norm = C21 / np.power(C21, 2)
        C40_norm = C40 / np.power(C21, 2)
        C41_norm = C41 / np.power(C21, 2)
        C42_norm = C42 / np.power(C21, 2)
        C60_norm = C60 / np.power(C21, 2)
        C61_norm = C61 / np.power(C21, 2)
        C62_norm = C62 / np.power(C21, 2)
        C63_norm = C63 / np.power(C21, 2)

        cumulants = np.zeros((1, 9), dtype=np.float64)
        cumulants[0, 0] = np.abs(C20_norm)
        cumulants[0, 1] = np.abs(C21_norm)
        cumulants[0, 2] = np.abs(C40_norm)
        cumulants[0, 3] = np.abs(C41_norm)
        cumulants[0, 4] = np.abs(C42_norm)
        cumulants[0, 5] = np.abs(C60_norm)
        cumulants[0, 6] = np.abs(C61_norm)
        cumulants[0, 7] = np.abs(C62_norm)
        cumulants[0, 8] = np.abs(C63_norm)

        return cumulants

    def load_cum(self, data_type, file_name):
        file_path = osp.join(
            self.data_root, 'sequence_data', data_type, file_name)
        seq_data = np.load(file_path)
        cum_data = self.generate_cumulants(seq_data)

        return cum_data

    def load_data(self, ann_file):
        """Load data from annotation file."""
        annos = IOLoad(ann_file)
        anno_data = annos['data']
        data = np.concatenate([self.load_cum('iq', item['filename']) for item in anno_data])
        label = np.array([item['ann']['labels'][0] for item in anno_data])

        # data = np.zeros((len(anno_data), 9), dtype=np.float64)
        # label = np.zeros(len(anno_data), dtype=np.int)
        # for item_index, item in enumerate(anno_data):
        #     iq_data = self.load_seq_data('iq', item['filename'])
        #     cu_fea = self.generate_cumulants(iq_data)
        #     data[item_index, :] = cu_fea
        #     label[item_index] = item['ann']['labels'][0]

        mods_dict = annos['mods']
        snrs_dict = annos['snrs']

        return data, label, anno_data, mods_dict, snrs_dict

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

        return self._parse_ann_info(ann_info_keys, ann_info_vals)

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

    def evaluate(self, results, prefix='common'):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        """
        confusion_matrix = np.zeros((len(self.SNRS), len(
            self.CLASSES), len(self.CLASSES)), dtype=np.float64)

        for idx in range(len(self)):
            ann = self.get_ann_info(idx)
            snrs = ann['snrs']
            labels = ann['mod_labels']
            if snrs.size == 1 and labels.size == 1:
                predict_class_index = results[idx]
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
        conf = np.sum(confusion_matrix, axis=0) / len(self.SNRS)
        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor
        eval_results[prefix + 'snr_mean_all'] = 1.0 * cor / (cor + ncor)
        return eval_results

    def format_out(self, out_dir, results):
        """Format the results to json and save.

        Args:
            out_dir (str): the out dir to save the json file
            results (numpy.ndarray]): Testing results of the
                dataset.
        """
        assert isinstance(results, np.ndarray), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
                format(len(results), len(self)))
        save_path = os.path.join(out_dir, 'pre.npy')
        save_results = np.zeros((len(results), len(self.CLASSES)), dtype=np.float64)
        for result_index, result in enumerate(results):
            save_results[result_index, result] = 1

        np.save(save_path, save_results)
        json_results = dict(CLASSES=self.CLASSES, SNRS=self.SNRS,
                            mods_dict=self.mods_dict, snrs_dict=self.snrs_dict, ANN=[])
        for idx in range(len(self)):
            ann = self.get_ann_info(idx)
            json_results['ANN'].append(ann)

        IODump(json_results, osp.join(out_dir, 'ann.json'))


if __name__ == '__main__':
    a = np.ones((2, 128))
    FBDataset.generate_cumulants(a)
