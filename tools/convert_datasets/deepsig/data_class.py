import copy
import json
import multiprocessing
import os
import os.path as osp
import pickle
import random
import sys
from concurrent import futures

import h5py
import numpy as np
from tqdm import tqdm

from csrr.datasets.utils import Constellation

_Constellation = Constellation()
CPU_COUNT = multiprocessing.cpu_count()


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar

    Args:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """

    format_str = "{0:." + str(decimals) + "f}"
    percents = format_str.format(100 * (iteration / float(total)))
    filledLength = int(round(bar_length * iteration / float(total)))
    bar = '' * filledLength + '-' * (bar_length - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' %
                     (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()


def generate_annotation(mod_to_label, snr_to_label, filter_config):
    label_to_mod = {label: mod for mod, label in mod_to_label.items()}
    label_to_snr = {label: snr for snr, label in snr_to_label.items()}
    annotations = {'filter_config': filter_config, 'item_filename': [],
                   'item_mod_label': [], 'item_snr_label': [], 'item_snr_index': [],
                   'item_snr_value': [], 'label_to_mod': label_to_mod, 'label_to_snr': label_to_snr,
                   'mod_to_label': mod_to_label, 'snr_to_label': snr_to_label, 'snr_to_index': snr_to_label,
                   }

    return annotations


def update_annotation(annotation, filename, snr, mod):
    mod_to_label = annotation['mod_to_label']
    snr_to_label = annotation['snr_to_label']
    snr_to_index = annotation['snr_to_index']

    annotation['item_snr_value'].append(snr)
    annotation['item_filename'].append(filename)
    annotation['item_mod_label'].append(mod_to_label[mod])
    annotation['item_snr_label'].append(snr_to_label['{:d}'.format(snr)])
    annotation['item_snr_index'].append(snr_to_index['{:d}'.format(snr)])

    return annotation


def combine_two_annotation(annotation1, annotation2):
    combine_annotation = copy.deepcopy(annotation1)
    update_list = ['item_snr_value', 'item_filename', 'item_mod_label', 'item_snr_label', 'item_snr_index']
    for key_name in update_list:
        combine_annotation[key_name].extend(copy.deepcopy(annotation2[key_name]))

    return combine_annotation


def save_seq_and_constellation_data(item, sequence_data_dir, constellation_data_dir):
    # Save sequence data of In-phase/Quadrature
    iq_dir = osp.join(sequence_data_dir, 'iq')
    if not osp.isdir(iq_dir):
        os.makedirs(iq_dir)
    iq_path = osp.join(iq_dir, item['filename'])
    np.save(iq_path, item['data'])

    # Save sequence data of Amplitude/Phase
    ap_dir = osp.join(sequence_data_dir, 'ap')
    if not osp.isdir(ap_dir):
        os.makedirs(ap_dir)
    ap_path = osp.join(ap_dir, item['filename'])
    data = item['data'][0, :] + 1j * item['data'][1, :]
    amplitude = np.abs(data)
    phase = np.angle(data)
    ap_data = np.vstack((amplitude, phase))
    np.save(ap_path, ap_data)

    # Save constellation data generated by different task parameters
    item_data = item['data']
    item_data[0, :] = item_data[0, :] / item['real_scale']
    item_data[1, :] = item_data[1, :] / item['imag_scale']
    constellations, filters = _Constellation.generate_by_filter(item_data)

    for constellation, param in zip(constellations, filters):
        constellation_dir = osp.join(constellation_data_dir,
                                     'filter_size_{:<.3f}_stride_{:<.3f}'.format(param[0], param[1]))
        if not osp.isdir(constellation_dir):
            os.makedirs(constellation_dir)
        constellation_path = osp.join(constellation_dir, item['filename'])
        np.save(constellation_path, constellation)


class DeepSigBase(object):
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

    def preprocess_original_data(self):
        print('Start converting data {}-{}'.format(self.name, self.version))
        raw_data_path = osp.join(self.data_dir, self.data_name)
        data = pickle.load(open(raw_data_path, 'rb'), encoding='bytes')
        snrs, mods = map(lambda j: sorted(
            list(set(map(lambda x: x[j], data.keys())))), [1, 0])

        dataset = []
        item_index = 0

        mod_to_label = {mod.decode('UTF-8'): index for index, mod in enumerate(mods)}
        update_mods_dict = dict()
        if hasattr(self, 'mod2mod'):
            for mod in mod_to_label:
                update_mods_dict[self.mod2mod[mod]] = mod_to_label[mod]
            mod_to_label = update_mods_dict

        snr_to_label = {'{:d}'.format(snr): index for index, snr in enumerate(snrs)}
        filter_config = [[0.02, 0.02], [0.05, 0.05]]

        test_annotation = generate_annotation(mod_to_label, snr_to_label, filter_config)
        train_annotation = generate_annotation(mod_to_label, snr_to_label, filter_config)
        validation_annotation = generate_annotation(mod_to_label, snr_to_label, filter_config)

        random.seed(0)

        mods_snrs = []
        for mod in mods:
            for snr in snrs:
                mods_snrs.append({'mod': mod, 'snr': snr})

        for item in tqdm(mods_snrs):
            mod = item['mod']
            snr = item['snr']
            item_num = data[(mod, snr)].shape[0]
            item_indices = [i for i in range(item_num)]
            random.shuffle(item_indices)

            if hasattr(self, 'mod2mod'):
                mod_ = self.mod2mod[mod.decode('UTF-8')]
            else:
                mod_ = mod.decode('UTF-8')

            train_indices = item_indices[:int(self.data_ratios[0] * item_num)]
            test_indices = item_indices[(int(sum(self.data_ratios[:2]) * item_num)):]

            for sub_item_index in item_indices:
                item = data[(mod, snr)][sub_item_index, :, :]
                item = item.astype(np.float64)
                filename = '{:0>12d}.npy'.format(item_index + sub_item_index)

                if sub_item_index in train_indices:
                    train_annotation = update_annotation(train_annotation, filename, snr, mod_)
                elif sub_item_index in test_indices:
                    test_annotation = update_annotation(test_annotation, filename, snr, mod_)
                else:
                    validation_annotation = update_annotation(validation_annotation, filename, snr, mod_)
                real_scale = np.max(
                    np.abs(item[0, :])) + np.finfo(np.float64).eps
                imag_scale = np.max(
                    np.abs(item[1, :])) + np.finfo(np.float64).eps
                dataset.append({'filename': filename, 'data': item,
                                'real_scale': real_scale, 'imag_scale': imag_scale})
            item_index += item_num

        return dataset, train_annotation, validation_annotation, test_annotation

    def generate(self):
        try:
            dataset, train_annotation, validation_annotation, test_annotation = self.preprocess_original_data()

            train_and_validation_annotation = combine_two_annotation(train_annotation, validation_annotation)

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
            json.dump(train_annotation,
                      open(self.data_dir + '/{}.json'.format('train'), 'w'),
                      indent=4, sort_keys=True)
            json.dump(validation_annotation,
                      open(self.data_dir + '/{}.json'.format('validation'), 'w'),
                      indent=4, sort_keys=True)
            json.dump(test_annotation,
                      open(self.data_dir + '/{}.json'.format('test'), 'w'),
                      indent=4, sort_keys=True)
            json.dump(train_and_validation_annotation,
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

        mod_to_label = {mod: index for index, mod in enumerate(DeepSigD.MODS)}
        snrs = [snr for snr in range(-20, 32, 2)]
        snr_to_label = {'{:d}'.format(snr): index for index, snr in enumerate(snrs)}
        filter_config = [[0.02, 0.02], [0.05, 0.05]]

        test_annotation = generate_annotation(mod_to_label, snr_to_label, filter_config)
        train_annotation = generate_annotation(mod_to_label, snr_to_label, filter_config)
        validation_annotation = generate_annotation(mod_to_label, snr_to_label, filter_config)

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

                filename = '{:0>12d}.npy'.format(item_index)
                mod_index = np.argmax(data_mods[item_index, :])

                if item_index in train_indices:
                    train_annotation = update_annotation(train_annotation, filename,
                                                         int(data_snrs[item_index, 0]), self.MODS[mod_index])
                elif item_index in test_indices:
                    test_annotation = update_annotation(test_annotation, filename,
                                                        int(data_snrs[item_index, 0]), self.MODS[mod_index])
                else:
                    validation_annotation = update_annotation(validation_annotation, filename,
                                                              int(data_snrs[item_index, 0]), self.MODS[mod_index])

                real_scale = np.max(np.abs(item[0, :])) + np.finfo(np.float64).eps
                imag_scale = np.max(np.abs(item[1, :])) + np.finfo(np.float64).eps

                dataset.append({'filename': filename, 'data': item,
                                'real_scale': real_scale, 'imag_scale': imag_scale})

        return dataset, train_annotation, validation_annotation, test_annotation
