import json
import multiprocessing
import os
import os.path as osp
import random
import sys
from abc import abstractmethod
from concurrent import futures

import h5py
import numpy as np

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


def generate_json(data, mods, snrs):
    res = {'data': data, 'mods': mods, 'snrs': snrs}

    return res


def save_data(item, data_dir):
    # Save data
    data_path = osp.join(data_dir, item['filename'])
    np.save(data_path, item['data'])


class WTIBase(object):
    CLASSES = [
        'BPSK',
        'QPSK',
        '8PSK',
        '16QAM',
        '32QAM',
        '64QAM',
        '128QAM',
        '256QAM',
    ]

    def __init__(self, root_dir, version):

        self.name = 'MATSLOT'
        self.root_dir = root_dir
        self.version = version
        self.data_dir = osp.join(self.root_dir, self.name, self.version)
        self.file_path = ''
        self.mode = 'train'

    @abstractmethod
    def preprocess_original_data(self):
        pass

    def set_mode(self, mode):
        self.mode = mode

    def set_file_path(self, file_path):
        self.file_path = file_path

    def generate(self):
        try:
            dataset, annotations, mods_dict, snrs_dict = self.preprocess_original_data()

            data_dir = osp.join(self.data_dir, 'data')

            if not osp.isdir(data_dir):
                os.makedirs(data_dir)

            # Save the item as *.npy file
            num_items = len(dataset)
            with futures.ProcessPoolExecutor(max_workers=CPU_COUNT) as executor:
                fs = [executor.submit(save_data, item, data_dir)
                      for item in dataset]
                for i, f in enumerate(futures.as_completed(fs)):
                    # Write progress to error so that it can be seen
                    print_progress(i, num_items, prefix='Convert {}-{}'.format(self.name, self.version), suffix='Done ',
                                   bar_length=40)

            data = generate_json(annotations, mods_dict, snrs_dict)

            print(
                'Save {} annotation json for the data set {}-{}'.format(self.mode, self.name, self.version))
            json.dump(data, open(
                self.data_dir + '/{}.json'.format(self.mode), 'w'), indent=4, sort_keys=True)
        except Exception as e:
            print('Error Message is: {}'.format(e))
            raise RuntimeError(
                'Converting data {}-{} failed'.format(self.name, self.version))


class Slot(WTIBase):
    def __init__(self, root_dir, version):
        super(Slot, self).__init__(root_dir, version)

    def preprocess_original_data(self):
        print('Start converting {} data {}-{}'.format(self.mode,
                                                      self.name, self.version))
        if isinstance(self.file_path, list):
            labels_list = []
            snrs_list = []
            slot_data_list = []
            for file_path in self.file_path:
                data = h5py.File(file_path, 'r')
                snrs = np.array(
                    data['slotted_CD_Output_SNR'], dtype=np.float32)
                labels = np.array(data['output'], dtype=np.int16)
                labels = np.transpose(labels, (1, 0))
                slot_data = np.array(
                    data['slotted_CD_reieve_CNN'], dtype=np.uint8)
                slot_data = np.transpose(slot_data, (3, 2, 0, 1))
                labels_list.append(labels)
                snrs_list.append(snrs)
                slot_data_list.append(slot_data)

            labels = np.concatenate(labels_list, axis=0)
            snrs = np.concatenate(snrs_list, axis=1)
            slot_data = np.concatenate(slot_data_list, axis=0)
        else:
            data = h5py.File(self.file_path, 'r')
            snrs = np.array(data['slotted_CD_Output_SNR'], dtype=np.float32)
            labels = np.array(data['output'], dtype=np.int16)
            labels = np.transpose(labels, (1, 0))
            slot_data = np.array(data['slotted_CD_reieve_CNN'], dtype=np.uint8)
            slot_data = np.transpose(slot_data, (3, 2, 0, 1))

        mods = self.CLASSES

        dataset = []
        annotations = []
        mods_dict = {mod: index for index,
                                    mod in enumerate(mods)}
        snrs_dict = {'{:.3f}'.format(
            snr): snr_index for snr_index, snr in enumerate(range(int(np.min(snrs[0])), int(np.max(snrs[0])) + 1, 1))}

        random.seed(0)

        for item_index in range(slot_data.shape[0]):
            if self.mode == 'train':
                filename = '0-{:0>12d}.npy'.format(item_index)
            elif self.mode == 'val':
                filename = '1-{:0>12d}.npy'.format(item_index)
            else:
                filename = '2-{:0>12d}.npy'.format(item_index)

            item = slot_data[item_index, :, :, :]
            mod = np.argmax(labels[item_index, :])
            snr = snrs[0, item_index]
            ann_info = {'labels': [int(mod)],
                        'snrs': [float(snr)]}

            annotation_item = {'filename': filename, 'ann': ann_info}
            annotations.append(annotation_item)
            dataset.append({'filename': filename, 'data': item})
            # if not osp.isfile(osp.join(self.data_dir, 'data', filename)):
            #     dataset.append({'filename': filename, 'data': item})

        return dataset, annotations, mods_dict, snrs_dict


class RawIQ(WTIBase):
    def __init__(self, root_dir, version):
        super(RawIQ, self).__init__(root_dir, version)

    def preprocess_original_data(self):
        print('Start converting {} data {}-{}'.format(self.mode,
                                                      self.name, self.version))
        if isinstance(self.file_path, list):
            labels_list = []
            snrs_list = []
            iq_data_list = []
            for file_path in self.file_path:
                data = h5py.File(file_path, 'r')
                snrs = np.array(data['IQ_Output_SNR'], dtype=np.float32)
                labels = np.array(data['output'], dtype=np.int16)
                labels = np.transpose(labels, (1, 0))
                iq_data = np.array(data['IQ_mulN_reieve_CNN'], dtype=np.float32)
                iq_data = np.transpose(iq_data, (2, 1, 0))
                labels_list.append(labels)
                snrs_list.append(snrs)
                iq_data_list.append(iq_data)

            labels = np.concatenate(labels_list, axis=0)
            snrs = np.concatenate(snrs_list, axis=1)
            iq_data = np.concatenate(iq_data_list, axis=0)
        else:
            data = h5py.File(self.file_path, 'r')
            snrs = np.array(data['IQ_Output_SNR'], dtype=np.float32)
            labels = np.array(data['output'], dtype=np.int16)
            labels = np.transpose(labels, (1, 0))
            iq_data = np.array(data['IQ_mulN_reieve_CNN'], dtype=np.float32)
            iq_data = np.transpose(iq_data, (2, 1, 0))

        mods = self.CLASSES

        dataset = []
        annotations = []
        mods_dict = {mod: index for index,
                                    mod in enumerate(mods)}
        snrs_dict = {'{:.3f}'.format(
            snr): snr_index for snr_index, snr in enumerate(range(int(np.min(snrs[0])), int(np.max(snrs[0])) + 1, 1))}

        random.seed(0)

        for item_index in range(iq_data.shape[0]):
            if self.mode == 'train':
                filename = '0-{:0>12d}.npy'.format(item_index)
            elif self.mode == 'val':
                filename = '1-{:0>12d}.npy'.format(item_index)
            else:
                filename = '2-{:0>12d}.npy'.format(item_index)

            item = iq_data[item_index, :, :]
            mod = np.argmax(labels[item_index, :])
            snr = snrs[0, item_index]
            ann_info = {'labels': [int(mod)],
                        'snrs': [float(snr)]}

            annotation_item = {'filename': filename, 'ann': ann_info}
            annotations.append(annotation_item)
            dataset.append({'filename': filename, 'data': item})

            # if not osp.isfile(osp.join(self.data_dir, 'data', filename)):
            #     dataset.append({'filename': filename, 'data': item})

        return dataset, annotations, mods_dict, snrs_dict
