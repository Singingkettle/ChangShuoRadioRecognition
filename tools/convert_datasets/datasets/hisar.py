import os.path as osp
import random

import numpy as np
import pandas as pd

from .base_dataset import BaseDataset, combine_two_infos


class HisarMod2019(BaseDataset):
    MODS = {
        0: 'BPSK',
        10: 'QPSK',
        20: '8PSK',
        30: '16PSK',
        40: '32PSK',
        50: '64PSK',
        1: '4QAM',
        11: '8QAM',
        21: '16QAM',
        31: '32QAM',
        41: '64QAM',
        51: '128QAM',
        61: '256QAM',
        2: '2FSK',
        12: '4FSK',
        22: '8FSK',
        32: '16FSK',
        3: '4PAM',
        13: '8PAM',
        23: '16PAM',
        4: 'AM-DSB',
        14: 'AM-DSB-SC',
        24: 'AM-USB',
        34: 'AM-LSB',
        44: 'FM',
        54: 'PM'
    }

    def __init__(self, root_dir, version, data_ratios=None):
        super(HisarMod2019, self).__init__('Hisar', root_dir, version, data_ratios)

    def preprocess_original_data(self):
        print(f'Start converting data {self.organization}-{self.version}')

        def load_csv(data_path):
            data = pd.read_csv(data_path, header=None, engine='pyarrow')
            return np.squeeze(np.array(data))

        test_snr = load_csv(osp.join(self.data_dir, self.data_name, 'Test', 'test_snr.csv'))
        test_label = load_csv(osp.join(self.data_dir, self.data_name, 'Test', 'test_labels.csv'))
        test_data = load_csv(osp.join(self.data_dir, self.data_name, 'Test', 'test_data.csv'))

        train_snr = load_csv(osp.join(self.data_dir, self.data_name, 'Train', 'train_snr.csv'))
        train_data = load_csv(osp.join(self.data_dir, self.data_name, 'Train', 'train_data.csv'))
        train_label = load_csv(osp.join(self.data_dir, self.data_name, 'Train', 'train_labels.csv'))

        modulations = list(self.MODS.values())
        snrs = np.sort(np.unique(train_snr[:])).tolist()

        test_info = self._generate_new_info(modulations, snrs)
        train_info = self._generate_new_info(modulations, snrs)
        validation_info = self._generate_new_info(modulations, snrs)

        random.seed(0)

        indices = []
        for index in range(test_data.shape[0]):
            indices.append(dict(data='test', index=index))

        for snr in snrs:
            for mod in np.sort(np.unique(train_label[:])):
                item_indices = ((train_label == mod) & (train_snr == snr)).nonzero()[0].tolist()
                random.shuffle(item_indices)
                train_indices = [dict(data='train', index=index) for index in
                                 item_indices[:int(0.8 * len(item_indices))]]
                validation_indices = [dict(data='validation', index=index) for index in
                                      item_indices[int(0.8 * len(item_indices)):]]
                indices.extend(train_indices)
                indices.extend(validation_indices)

        dataset = []
        for item_index, item in enumerate(indices):
            file_name = '{:0>12d}.npy'.format(item_index)

            if item['data'] == 'test':
                snr = int(test_snr[item['index']])
                item_data = test_data[item['index'], :]
                modulation = self.MODS[test_label[item['index']]]
                test_info['data_list'].append(dict(file_name=file_name, snr=snr, modulation=modulation))
            elif item['data'] == 'train':
                snr = int(train_snr[item['index']])
                item_data = train_data[item['index'], :]
                modulation = self.MODS[train_label[item['index']]]
                train_info['data_list'].append(dict(file_name=file_name, snr=snr, modulation=modulation))
            else:
                snr = int(train_snr[item['index']])
                item_data = train_data[item['index'], :]
                modulation = self.MODS[train_label[item['index']]]
                validation_info['data_list'].append(dict(file_name=file_name, snr=snr, modulation=modulation))
            item_data = np.char.replace(item_data.astype(str), 'i', 'j').astype(np.complex128)
            item_data = np.vstack([np.real(item_data), np.imag(item_data)])
            dataset.append(dict(file_name=file_name, data=item_data))

        train_and_validation_info = combine_two_infos(train_info, validation_info)

        infos = dict(train=train_info, validation=validation_info,
                     test=test_info, train_and_validation=train_and_validation_info)

        return dataset, infos


if __name__ == '__main__':
    dataset = HisarMod2019('/home/citybuster/Data/SignalProcessing/ModulationClassification', 'HisarMod2019.1')
    a = dataset.preprocess_original_data
