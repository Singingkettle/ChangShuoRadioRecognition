import json
import multiprocessing
import os
import os.path as osp
import sys
from concurrent import futures

import h5py
import numpy as np

from .constellation import Constellation

CPU_COUNT = multiprocessing.cpu_count()


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    '''
    Call in a loop to create terminal progress bar

    Args:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    '''

    format_str = '{0:.' + str(decimals) + 'f}'
    percents = format_str.format(100 * (iteration / float(total)))
    filledLength = int(round(bar_length * iteration / float(total)))
    bar = '' * filledLength + '-' * (bar_length - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' %
                     (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()


def generate_json(data, mods, snrs, filters):
    res = {'data': data, 'mods': mods, 'snrs': snrs, 'filters': filters}

    return res


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
    constellations, filters = Constellation.generate_by_filter(item_data)

    for constellation, param in zip(constellations, filters):
        constellation_dir = osp.join(constellation_data_dir,
                                     'filter_size_{:<.3f}_stride_{:<.3f}'.format(param[0], param[1]))
        if not osp.isdir(constellation_dir):
            os.makedirs(constellation_dir)
        constellation_path = osp.join(constellation_dir, item['filename'])
        np.save(constellation_path, constellation)


class MatlabData(object):
    CLASSES = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', 'PAM4', 'GFSK', 'CPFSK', 'B-FM', 'DSB-AM', 'SSB-AM']

    def __init__(self, root_dir, version, data_ratios):

        self.name = 'Matlab'
        self.root_dir = root_dir
        self.version = version
        self.data_dir = osp.join(self.root_dir, self.name, self.version)
        self.train_num = 0
        self.val_num = 0
        self.test_num = 0
        self.data_ratios = data_ratios
        self.filters = Constellation.get_filters()
        self.mat_list = ['train.mat', 'test.mat', 'val.mat']
        self.mods = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', 'PAM4', 'GFSK', 'CPFSK', 'B-FM', 'DSB-AM', 'SSB-AM']

    def load_mat_data(self, mat_path, item_index):
        with h5py.File(mat_path, 'r') as f:
            iq = np.array(f['data']['iq'])
            iq = np.transpose(iq, (4, 3, 2, 1, 0))
            snrs = np.array(f['data']['SNRs'])

        index_snrs = [snr for snr in snrs[10:iq.shape[1] - 10:2, 0]]
        mods_dict = {mod: index for index, mod in enumerate(self.mods)}
        snrs_dict = {'{:.3f}'.format(snr): index for index, snr in enumerate(index_snrs)}

        annotations = []
        dataset = []
        for mod_index in range(iq.shape[0]):
            for snr_index in range(10, iq.shape[1] - 10, 2):
                for select_item_index in range(iq.shape[2]):  # only include -10dB to 20 dB
                    item = iq[mod_index, snr_index, select_item_index, :, :]
                    item = item.astype(np.float64)
                    item_index += 1
                    filename = '{:0>12d}.npy'.format(item_index)

                    ann_info = {'labels': [mod_index],
                                'snrs': [float(snrs[snr_index, 0])]}

                    annotation_item = {'filename': filename, 'ann': ann_info}
                    annotations.append(annotation_item)
                    real_scale = np.max(
                        np.abs(item[0, :])) + np.finfo(np.float64).eps
                    imag_scale = np.max(
                        np.abs(item[1, :])) + np.finfo(np.float64).eps
                    if (not osp.isfile(osp.join(self.data_dir, 'sequence_data', 'iq', filename))) or (
                            not osp.isfile(osp.join(self.data_dir, 'sequence_data', 'ap', filename))):
                        dataset.append({'filename': filename, 'data': item,
                                        'real_scale': real_scale, 'imag_scale': imag_scale})

        return dataset, annotations, mods_dict, snrs_dict, item_index

    def preprocess_original_data(self):
        print('Start converting data {}-{}'.format(self.name, self.version))

        dataset = []
        val_dataset, val_annotations, mods_dict, snrs_dict, item_index = self.load_mat_data(
            os.path.join(self.data_dir, 'val.mat'), 0)
        test_dataset, test_annotations, mods_dict, snrs_dict, item_index = self.load_mat_data(
            os.path.join(self.data_dir, 'test.mat'), item_index)
        train_dataset, train_annotations, mods_dict, snrs_dict, item_index = self.load_mat_data(
            os.path.join(self.data_dir, 'train.mat'), item_index)
        dataset.extend(train_dataset)
        dataset.extend(val_dataset)
        dataset.extend(test_dataset)

        return dataset, train_annotations, val_annotations, test_annotations, mods_dict, snrs_dict

    def generate(self):
        try:
            dataset, train_annotations, val_annotations, test_annotations, mods_dict, snrs_dict = self.preprocess_original_data()

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

            train_data = generate_json(
                train_annotations, mods_dict, snrs_dict, self.filters)
            val_data = generate_json(
                val_annotations, mods_dict, snrs_dict, self.filters)
            test_data = generate_json(
                test_annotations, mods_dict, snrs_dict, self.filters)
            train_and_val_data = generate_json(
                train_annotations + val_annotations, mods_dict, snrs_dict, self.filters)

            print(
                'Save train, val, test annotation json for the data set {}-{}'.format(self.name, self.version))
            json.dump(train_data, open(self.data_dir + '/{}.json'.format('train'), 'w'),
                      indent=4, sort_keys=True)
            json.dump(val_data, open(self.data_dir + '/{}.json'.format('val'), 'w'),
                      indent=4, sort_keys=True)
            json.dump(test_data, open(self.data_dir + '/{}.json'.format('test'), 'w'),
                      indent=4, sort_keys=True)
            json.dump(train_and_val_data, open(self.data_dir + '/{}.json'.format('train_and_val'), 'w'),
                      indent=4, sort_keys=True)
        except Exception as e:
            print('Error Message is: {}'.format(e))
            raise RuntimeError(
                'Converting data {}-{} failed'.format(self.name, self.version))
