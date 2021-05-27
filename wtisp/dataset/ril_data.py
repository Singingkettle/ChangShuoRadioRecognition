import os.path as osp

import numpy as np
import scipy.io
from torch.utils.data import Dataset

from .builder import DATASETS
from ..common.fileio import dump as IODump
from ..common.fileio import load as IOLoad


@DATASETS.register_module()
class WTIRILDataset(Dataset):

    def __init__(self, ann_file, data_root=None, test_mode=False, error_threshold=8.4):

        self.ann_file = ann_file
        self.data_root = data_root
        self.test_mode = test_mode
        self.error_threshold = error_threshold

        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)

        # load annotations
        self.data_infos, self.anchor_xs, self.anchor_ys = self.load_annotations(
            self.ann_file)

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        annos = IOLoad(ann_file)
        anno_data = annos['data']
        xs = annos['labelx']
        ys = annos['labely']
        xs = np.array(xs, dtype=np.float32)
        ys = np.array(ys, dtype=np.float32)

        def convert_position_as_probability_distribution(anchors, query):
            gt_probability_distribution = np.zeros(
                anchors.shape[0], dtype=np.float32)

            if query <= anchors[0]:
                gt_probability_distribution[0] = 1.0
                return gt_probability_distribution
            elif query >= anchors[-1]:
                gt_probability_distribution[-1] = 1.0
            else:
                # https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html
                insert_index = np.searchsorted(anchors, query, side='right')
                if query == anchors[insert_index - 1]:
                    gt_probability_distribution[insert_index - 1] = 1.0
                else:
                    d1 = query - anchors[insert_index - 1]
                    d2 = anchors[insert_index] - query
                    p1 = d2 / (d1 + d2)
                    p2 = d1 / (d1 + d2)
                    gt_probability_distribution[insert_index] = p2
                    gt_probability_distribution[insert_index - 1] = p1

            return gt_probability_distribution

        tmp_anno_data = []
        for item in anno_data:
            tmp_item = dict()
            tmp_item['filename'] = item['filename']
            tmp_item['x'] = item['labelx']
            tmp_item['y'] = item['labely']
            tmp_item['px'] = convert_position_as_probability_distribution(
                xs, item['labelx'])
            tmp_item['py'] = convert_position_as_probability_distribution(
                ys, item['labely'])
            tmp_anno_data.append(tmp_item)

        anno_data = tmp_anno_data

        return anno_data, xs, ys

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
        px = self.data_infos[idx]['px']
        py = self.data_infos[idx]['py']

        return dict(px=px, py=py)

    def prepare_train_data(self, idx):
        """Get training data.

        Args.
         idx (int): Index of data

        Returns:
            dict: Training data and annotation
        """

        item_info = self.data_infos[idx]

        file_name = item_info['filename']

        data_path = osp.join(self.data_root, file_name)
        x = scipy.io.loadmat(data_path)
        x = x['c']
        x = x.astype(np.float32)
        x = np.reshape(x, (1, 8, -1))
        x = x[:, :, :1024]
        ann = self.get_ann_info(idx)

        data = dict(x=x, px=ann['px'], py=ann['py'])

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

        data_path = osp.join(self.data_root, file_name)
        x = scipy.io.loadmat(data_path)
        x = x['c']
        x = x.astype(np.float32)
        x = np.reshape(x, (1, 8, -1))
        x = x[:, :, :1024]

        data = dict(x=x)

        return data

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
        eval_results = dict()

        delta_xs = np.zeros(len(results), dtype=np.float32)
        delta_ys = np.zeros(len(results), dtype=np.float32)
        delta_os = np.zeros(len(results), dtype=np.float32)
        for idx, item in enumerate(results):
            e_x = np.dot(item['px'], self.anchor_xs)
            e_y = np.dot(item['py'], self.anchor_ys)
            gt_x = self.data_infos[idx]['x']
            gt_y = self.data_infos[idx]['y']
            delta_xs[idx] = abs(gt_x - e_x)
            delta_ys[idx] = abs(gt_y - e_y)
            delta_os[idx] = (delta_xs[idx] ** 2 + delta_ys[idx] ** 2) ** (1 / 2)

        eval_results['delta_x_min'] = np.min(delta_xs)
        eval_results['delta_x_max'] = np.max(delta_xs)
        eval_results['delta_y_min'] = np.min(delta_ys)
        eval_results['delta_y_max'] = np.max(delta_ys)
        eval_results['delta_o_min'] = np.min(delta_os)
        eval_results['delta_o_max'] = np.max(delta_os)

        sorted_delta_os = np.sort(delta_os)
        eval_results['error_0.8'] = sorted_delta_os[int(len(results) * 0.8) - 1]
        eval_results['positive_radio'] = (
                                                 delta_os < self.error_threshold).sum() / len(results)

        return eval_results
