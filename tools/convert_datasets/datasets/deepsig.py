import copy
import multiprocessing
import os.path as osp
import pickle
import random

import h5py
import numpy as np
from tqdm import tqdm

from .base_dataset import BaseDataset, Constellation, combine_two_infos

_Constellation = Constellation()
CPU_COUNT = multiprocessing.cpu_count()


class DeepSigBase(BaseDataset):
    def __init__(self, root_dir, version, data_ratios):
        super(DeepSigBase, self).__init__('DeepSig', root_dir, version, data_ratios)

    def preprocess_original_data(self):
        print('Start converting data {}-{}'.format(self.organization, self.version))
        raw_data_path = osp.join(self.data_dir, self.data_name)
        data = pickle.load(open(raw_data_path, 'rb'), encoding='bytes')
        rsnrs, rmods = map(lambda j: sorted(list(set(map(lambda x: x[j], data.keys())))), [1, 0])

        dataset = []
        item_index = 0

        modulations = []
        for mod in copy.deepcopy(rmods):
            if hasattr(mod, 'decode'):
                mod = mod.decode('UTF-8')
            modulations.append(mod)

        if hasattr(self, 'mod2mod'):
            modulations = [self.mod2mod[mod] for mod in modulations]

        snrs = [int(snr) for snr in rsnrs]

        test_info = dict(filter_config=self.filter_config, modulations=modulations, snrs=snrs, annotations=[])
        train_info = dict(filter_config=self.filter_config, modulations=modulations, snrs=snrs, annotations=[])
        validation_info = dict(filter_config=self.filter_config, modulations=modulations, snrs=snrs, annotations=[])

        random.seed(0)
        for bmodulation, snr in tqdm(data.keys()):
            sub_data = data[(bmodulation, snr)]
            item_num = sub_data.shape[0]
            item_indices = [i for i in range(item_num)]

            random.shuffle(item_indices)

            if hasattr(bmodulation, 'decode'):
                modulation = copy.deepcopy(bmodulation.decode('UTF-8'))
            else:
                modulation = copy.deepcopy(bmodulation)

            if hasattr(self, 'mod2mod'):
                modulation = self.mod2mod[modulation]
            else:
                modulation = modulation

            train_indices = item_indices[:int(self.data_ratios[0] * item_num)]
            test_indices = item_indices[(int(sum(self.data_ratios[:2]) * item_num)):]

            for sub_item_index in item_indices:
                item_data = sub_data[sub_item_index, :, :]
                item_data = item_data.astype(np.float64)
                file_name = '{:0>12d}.npy'.format(item_index + sub_item_index)
                if sub_item_index in train_indices:
                    train_info['annotations'].append(dict(file_name=file_name, snr=snr, modulation=modulation))
                elif sub_item_index in test_indices:
                    test_info['annotations'].append(dict(file_name=file_name, snr=snr, modulation=modulation))
                else:
                    validation_info['annotations'].append(dict(file_name=file_name, snr=snr, modulation=modulation))
                real_scale = np.max(np.abs(item_data[0, :])) + np.finfo(np.float64).eps
                imag_scale = np.max(np.abs(item_data[1, :])) + np.finfo(np.float64).eps
                dataset.append({'file_name': file_name, 'data': item_data,
                                'real_scale': real_scale, 'imag_scale': imag_scale})
            item_index += item_num

        train_and_validation_info = combine_two_infos(train_info, validation_info)

        infos = dict(train=train_info, validation=validation_info,
                     test=test_info, train_and_validation=train_and_validation_info)

        return dataset, infos


class DeepSigA(DeepSigBase):
    def __init__(self, root_dir, version, data_ratios):
        super(DeepSigA, self).__init__(root_dir, version, data_ratios)
        self.data_name = '2016.04C.multisnr.pkl'
        self.mod2mod = {'8PSK': '8PSK', 'AM-DSB': 'AM-DSB', 'AM-SSB': 'AM-SSB', 'BPSK': 'BPSK', 'CPFSK': 'CPFSK',
                        'GFSK': 'GFSK', 'PAM4': '4PAM', 'QAM16': '16QAM', 'QAM64': '64QAM', 'QPSK': 'QPSK',
                        'WBFM': 'WBFM'}


class DeepSigB(DeepSigBase):
    def __init__(self, root_dir, version, data_ratios):
        super(DeepSigB, self).__init__(root_dir, version, data_ratios)
        self.data_name = 'RML2016.10a_dict.pkl'
        self.mod2mod = {'8PSK': '8PSK', 'AM-DSB': 'AM-DSB', 'AM-SSB': 'AM-SSB', 'BPSK': 'BPSK', 'CPFSK': 'CPFSK',
                        'GFSK': 'GFSK', 'PAM4': '4PAM', 'QAM16': '16QAM', 'QAM64': '64QAM', 'QPSK': 'QPSK',
                        'WBFM': 'WBFM'}


class DeepSigC(DeepSigBase):
    def __init__(self, root_dir, version, data_ratios):
        super(DeepSigC, self).__init__(root_dir, version, data_ratios)
        self.data_name = 'RML2016.10b.dat'
        self.mod2mod = {'8PSK': '8PSK', 'AM-DSB': 'AM-DSB', 'AM-SSB': 'AM-SSB', 'BPSK': 'BPSK', 'CPFSK': 'CPFSK',
                        'GFSK': 'GFSK', 'PAM4': '4PAM', 'QAM16': '16QAM', 'QAM64': '64QAM', 'QPSK': 'QPSK',
                        'WBFM': 'WBFM'}


class DeepSigD(DeepSigBase):
    # The details of MODS are inferred from the paper 'Over-the-Air Deep Learning Based Radio Signal Classification',
    # which are verified by signal constellation. The class.txt is wrong.
    # also in these links, the same issue is mentioned
    # https://github.com/radioML/dataset/issues/25
    # https://blog.csdn.net/weixin_40692714/article/details/120434505
    # https://blog.csdn.net/weixin_43663595/article/details/112580100
    MODS = ['OOK',
            '4ASK',
            '8ASK',
            'BPSK',
            'QPSK',
            '8PSK',
            '16PSK',
            '32PSK',
            '16APSK',
            '32APSK',
            '64APSK',
            '128APSK',
            '16QAM',
            '32QAM',
            '64QAM',
            '128QAM',
            '256QAM',
            'AM-SSB-WC',
            'AM-SSB-SC',
            'AM-DSB-WC',
            'AM-DSB-SC',
            'FM',
            'GMSK',
            'OQPSK']

    def __init__(self, root_dir, version, data_ratios):
        super(DeepSigD, self).__init__(root_dir, version, data_ratios)
        self.data_name = 'GOLD_XYZ_OSC.0001_1024.hdf5'

    def preprocess_original_data(self):
        print(f'Start converting data {self.organization}-{self.version}')
        raw_data_path = osp.join(self.data_dir, self.data_name)
        data = h5py.File(raw_data_path, 'r')

        sequence_data = data['X']
        data_mods = data['Y']
        data_snrs = data['Z']

        sequence_data = sequence_data[:, :, :]
        sequence_data = np.transpose(sequence_data, (0, 2, 1))

        data_map = dict()
        for item_index in tqdm(range(data_mods.shape[0])):
            mod_index = np.argmax(data_mods[item_index, :])
            key_str = '{}-{}'.format(mod_index,
                                     float(data_snrs[item_index, 0]))
            if key_str not in data_map:
                data_map[key_str] = []
            data_map[key_str].append(item_index)

        dataset = []

        modulations = [mod for mod in DeepSigD.MODS]
        snrs = [snr for snr in range(-20, 32, 2)]

        test_info = dict(filter_config=self.filter_config, modulations=modulations, snrs=snrs, annotations=[])
        train_info = dict(filter_config=self.filter_config, modulations=modulations, snrs=snrs, annotations=[])
        validation_info = dict(filter_config=self.filter_config, modulations=modulations, snrs=snrs, annotations=[])

        random.seed(0)
        for tuple_key in data_map.keys():
            item_num = len(data_map[tuple_key])
            item_indices = [i for i in data_map[tuple_key]]
            random.shuffle(item_indices)

            train_indices = item_indices[:int(self.data_ratios[0] * item_num)]
            test_indices = item_indices[(int(sum(self.data_ratios[:2]) * item_num)):]
            for item_index in data_map[tuple_key]:
                item_data = sequence_data[item_index, :, :]
                item_data = item_data.astype(np.float64)

                file_name = '{:0>12d}.npy'.format(item_index)
                mod_index = np.argmax(data_mods[item_index, :])

                if item_index in train_indices:
                    train_info['annotations'].append(
                        dict(file_name=file_name, snr=int(data_snrs[item_index, 0]), modulation=self.MODS[mod_index]))
                elif item_index in test_indices:
                    test_info['annotations'].append(
                        dict(file_name=file_name, snr=int(data_snrs[item_index, 0]), modulation=self.MODS[mod_index]))
                else:
                    validation_info['annotations'].append(
                        dict(file_name=file_name, snr=int(data_snrs[item_index, 0]), modulation=self.MODS[mod_index]))

                real_scale = np.max(np.abs(item_data[0, :])) + np.finfo(np.float64).eps
                imag_scale = np.max(np.abs(item_data[1, :])) + np.finfo(np.float64).eps

                dataset.append({'file_name': file_name, 'data': item_data,
                                'real_scale': real_scale, 'imag_scale': imag_scale})

        train_and_validation_info = combine_two_infos(train_info, validation_info)

        infos = dict(train=train_info, validation=validation_info,
                     test=test_info, train_and_validation=train_and_validation_info)

        return dataset, infos
