import json
import multiprocessing
import os
import os.path as osp
import pickle
import random
from concurrent import futures

import h5py
import numpy as np
from tools.convert_datasets.common import print_progress, init_annotations, update_annotations, combine_two_annotations, \
    save_seq_and_constellation_data
from tqdm import tqdm

from csrr.datasets.pipeline.loading import Constellation

_Constellation = Constellation()
CPU_COUNT = multiprocessing.cpu_count()


class DeepSigBase:
    def __init__(self, root_dir, version, data_ratios):

        self.name = 'DeepSig'
        self.root_dir = root_dir
        self.version = version
        self.data_dir = osp.join(self.root_dir, self.name, self.version)
        self.train_num = 0
        self.val_num = 0
        self.test_num = 0
        self.data_ratios = data_ratios
        self.data_name = ''
        co = Constellation()
        self.filters = co.get_filters()

    @property
    def preprocess_original_data(self):
        print('Start converting data {}-{}'.format(self.name, self.version))
        raw_data_path = osp.join(self.data_dir, self.data_name)
        data = pickle.load(open(raw_data_path, 'rb'), encoding='bytes')
        rsnrs, rmods = map(lambda j: sorted(list(set(map(lambda x: x[j], data.keys())))), [1, 0])

        dataset = []
        item_index = 0

        modulations = []
        for mod in rmods:
            if hasattr(mod, 'decode'):
                mod = mod.decode('UTF-8')
            modulations.append(mod)

        if hasattr(self, 'mod2mod'):
            modulations = [self.mod2mod[mod] for mod in modulations]

        snrs = [int(snr) for snr in rsnrs]
        filter_config = [[0.02, 0.02], [0.05, 0.05]]

        test_annotations = init_annotations(modulations, snrs, filter_config)
        train_annotations = init_annotations(modulations, snrs, filter_config)
        validation_annotations = init_annotations(modulations, snrs, filter_config)

        random.seed(0)

        mods_snrs = []
        for mod in rmods:
            for snr in rsnrs:
                mods_snrs.append({'mod': mod, 'snr': snr})

        for item in tqdm(mods_snrs):
            mod = item['mod']
            snr = item['snr']
            item_num = data[(mod, snr)].shape[0]
            item_indices = [i for i in range(item_num)]
            random.shuffle(item_indices)

            if hasattr(mod, 'decode'):
                mod = mod.decode('UTF-8')

            if hasattr(self, 'mod2mod'):
                modulation = self.mod2mod[mod]
            else:
                modulation = mod

            train_indices = item_indices[:int(self.data_ratios[0] * item_num)]
            test_indices = item_indices[(int(sum(self.data_ratios[:2]) * item_num)):]

            for sub_item_index in item_indices:
                item = data[(mod, snr)][sub_item_index, :, :]
                item = item.astype(np.float64)
                file_name = '{:0>12d}.npy'.format(item_index + sub_item_index)

                if sub_item_index in train_indices:
                    train_annotations = update_annotations(train_annotations, file_name, snr, modulation)
                elif sub_item_index in test_indices:
                    test_annotations = update_annotations(test_annotations, file_name, snr, modulation)
                else:
                    validation_annotations = update_annotations(validation_annotations, file_name, snr, modulation)
                real_scale = np.max(
                    np.abs(item[0, :])) + np.finfo(np.float64).eps
                imag_scale = np.max(
                    np.abs(item[1, :])) + np.finfo(np.float64).eps
                dataset.append({'file_name': file_name, 'data': item,
                                'real_scale': real_scale, 'imag_scale': imag_scale})
            item_index += item_num

        return dataset, train_annotations, validation_annotations, test_annotations

    def generate(self):
        try:
            dataset, train_annotations, validation_annotations, test_annotations = self.preprocess_original_data

            train_and_validation_annotations = combine_two_annotations(train_annotations, validation_annotations)

            sequence_data_dir = osp.join(self.data_dir, 'sequence_data')
            constellation_data_dir = osp.join(
                self.data_dir, 'constellation_data')

            if not osp.isdir(sequence_data_dir):
                os.makedirs(sequence_data_dir)

            if not osp.isdir(constellation_data_dir):
                os.makedirs(constellation_data_dir)

            # Save the item as *.npy file
            num_items = len(dataset)
            with futures.ProcessPoolExecutor(max_workers=CPU_COUNT) as executor:
                fs = [executor.submit(save_seq_and_constellation_data, item, sequence_data_dir, constellation_data_dir)
                      for item in dataset]
                for i, f in enumerate(futures.as_completed(fs)):
                    # Write progress to error so that it can be seen
                    print_progress(i, num_items, prefix='Convert {}-{}'.format(self.name, self.version), suffix='Done ',
                                   bar_length=40)

            print('\nSave train, val, test annotation json for the data set {}-{}'.format(self.name, self.version))
            json.dump(train_annotations,
                      open(self.data_dir + '/{}.json'.format('train'), 'w'),
                      indent=4, sort_keys=True)
            json.dump(validation_annotations,
                      open(self.data_dir + '/{}.json'.format('validation'), 'w'),
                      indent=4, sort_keys=True)
            json.dump(test_annotations,
                      open(self.data_dir + '/{}.json'.format('test'), 'w'),
                      indent=4, sort_keys=True)
            json.dump(train_and_validation_annotations,
                      open(self.data_dir + '/{}.json'.format('train_and_validation'), 'w'),
                      indent=4, sort_keys=True)
        except Exception as e:
            print('Error Message is: {}'.format(e))
            raise RuntimeError(
                'Converting data {}-{} failed'.format(self.name, self.version))


class DeepSigA(DeepSigBase):
    def __init__(self, root_dir, version, data_ratios):
        super(DeepSigA, self).__init__(root_dir, version, data_ratios)
        self.data_name = '2016.04C.multisnr.pkl'
        self.mod2mod = {"8PSK": '8PSK', "AM-DSB": "AM-DSB", "AM-SSB": "AM-SSB", "BPSK": "BPSK", "CPFSK": "CPFSK",
                        "GFSK": "GFSK", "PAM4": "4PAM", "QAM16": "16QAM", "QAM64": "64QAM", "QPSK": "QPSK",
                        "WBFM": "WBFM"}


class DeepSigB(DeepSigBase):
    def __init__(self, root_dir, version, data_ratios):
        super(DeepSigB, self).__init__(root_dir, version, data_ratios)
        self.data_name = 'RML2016.10a_dict.pkl'
        self.mod2mod = {"8PSK": '8PSK', "AM-DSB": "AM-DSB", "AM-SSB": "AM-SSB", "BPSK": "BPSK", "CPFSK": "CPFSK",
                        "GFSK": "GFSK", "PAM4": "4PAM", "QAM16": "16QAM", "QAM64": "64QAM", "QPSK": "QPSK",
                        "WBFM": "WBFM"}


class DeepSigC(DeepSigBase):
    def __init__(self, root_dir, version, data_ratios):
        super(DeepSigC, self).__init__(root_dir, version, data_ratios)
        self.data_name = 'RML2016.10b.dat'
        self.mod2mod = {"8PSK": '8PSK', "AM-DSB": "AM-DSB", "AM-SSB": "AM-SSB", "BPSK": "BPSK", "CPFSK": "CPFSK",
                        "GFSK": "GFSK", "PAM4": "4PAM", "QAM16": "16QAM", "QAM64": "64QAM", "QPSK": "QPSK",
                        "WBFM": "WBFM"}


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

    @property
    def preprocess_original_data(self):
        print('Start converting data {}-{}'.format(self.name, self.version))
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
            key_str = "{}-{}".format(mod_index,
                                     float(data_snrs[item_index, 0]))
            if key_str not in data_map:
                data_map[key_str] = []
            data_map[key_str].append(item_index)

        dataset = []

        modulations = [mod for mod in DeepSigD.MODS]
        snrs = [snr for snr in range(-20, 32, 2)]
        filter_config = [[0.02, 0.02], [0.05, 0.05]]

        test_annotations = init_annotations(modulations, snrs, filter_config)
        train_annotations = init_annotations(modulations, snrs, filter_config)
        validation_annotations = init_annotations(modulations, snrs, filter_config)

        random.seed(0)
        for tuple_key in data_map.keys():
            item_num = len(data_map[tuple_key])
            item_indices = [i for i in data_map[tuple_key]]
            random.shuffle(item_indices)

            train_indices = item_indices[:int(self.data_ratios[0] * item_num)]
            test_indices = item_indices[(int(sum(self.data_ratios[:2]) * item_num)):]
            for item_index in data_map[tuple_key]:
                item = sequence_data[item_index, :, :]
                item = item.astype(np.float64)

                file_name = '{:0>12d}.npy'.format(item_index)
                mod_index = np.argmax(data_mods[item_index, :])

                if item_index in train_indices:
                    train_annotations = update_annotations(train_annotations, file_name,
                                                           int(data_snrs[item_index, 0]), self.MODS[mod_index])
                elif item_index in test_indices:
                    test_annotations = update_annotations(test_annotations, file_name,
                                                          int(data_snrs[item_index, 0]), self.MODS[mod_index])
                else:
                    validation_annotations = update_annotations(validation_annotations, file_name,
                                                                int(data_snrs[item_index, 0]), self.MODS[mod_index])

                real_scale = np.max(np.abs(item[0, :])) + np.finfo(np.float64).eps
                imag_scale = np.max(np.abs(item[1, :])) + np.finfo(np.float64).eps

                dataset.append({'file_name': file_name, 'data': item,
                                'real_scale': real_scale, 'imag_scale': imag_scale})

        return dataset, train_annotations, validation_annotations, test_annotations
