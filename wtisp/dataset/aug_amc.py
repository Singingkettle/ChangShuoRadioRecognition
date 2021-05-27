import os.path as osp

import numpy as np

from .amc_data import WTIMCDataset
from .builder import DATASETS


@DATASETS.register_module()
class AUGMCDataset(WTIMCDataset):

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
            step_data = np.arange(0, iq_data.shape[1], dtype=np.float32)
            iq_data = np.vstack((iq_data, step_data))
            iq_data = np.reshape(iq_data, (1, 3, -1))
            data['iqs'] = iq_data

        if self.ap:
            ap_path = osp.join(
                self.data_root, 'sequence_data', 'ap', file_name)
            ap_data = np.load(ap_path)
            ap_data = ap_data.astype(np.float32)
            step_data = np.arange(0, ap_data.shape[1], dtype=np.float32)
            ap_data = np.vstack((ap_data, step_data))
            ap_data = np.reshape(ap_data, (1, 3, -1))
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
            step_data = np.arange(0, iq_data.shape[1], dtype=np.float32)
            iq_data = np.vstack((iq_data, step_data))
            iq_data = np.reshape(iq_data, (1, 3, -1))
            data['iqs'] = iq_data

        if self.ap:
            ap_path = osp.join(
                self.data_root, 'sequence_data', 'ap', file_name)
            ap_data = np.load(ap_path)
            ap_data = ap_data.astype(np.float32)
            step_data = np.arange(0, ap_data.shape[1], dtype=np.float32)
            ap_data = np.vstack((ap_data, step_data))
            ap_data = np.reshape(ap_data, (1, 3, -1))
            data['aps'] = ap_data

        if self.co:
            co_path = osp.join(self.data_root, 'constellation_data',
                               'filter_size_{:<.3f}_stride_{:<.3f}'.format(self.filter[0], self.filter[1]), file_name)
            co_data = np.load(co_path)
            co_data = co_data.astype(np.float32)
            co_data = np.expand_dims(co_data, axis=0)
            data['cos'] = co_data

        return data
