import json
import multiprocessing
import os.path as osp
import random

from tools.convert_datasets.common import combine_two_annotations
from tqdm import tqdm

from csrr import glob
from csrr.common.fileio import load as IoLoad

CPU_COUNT = multiprocessing.cpu_count()


class CSRRData:
    def __init__(self, root_dir, version, data_ratios):
        self.name = 'ChangShuo'
        self.root_dir = root_dir
        self.version = version
        self.data_dir = osp.join(self.root_dir, self.name, self.version)
        self.anno_dir = osp.join(self.root_dir, self.name, self.version, 'anno')
        self.train_num = 0
        self.val_num = 0
        self.test_num = 0
        self.data_ratios = data_ratios
        self.data_name = ''

    @property
    def preprocess_original_data(self):

        jsons = glob(self.anno_dir, '.json')
        item_num = len(jsons)
        random.seed(0)
        item_indices = [i for i in range(len(jsons))]
        random.shuffle(item_indices)

        train_indices = item_indices[:int(self.data_ratios[0] * item_num)]
        test_indices = item_indices[(int(sum(self.data_ratios[:2]) * item_num)):]

        test_annotations = dict(annotations=[], modulations=[], snrs=[], channels=[], sample_rate=150e3,
                                sample_num=1200)
        train_annotations = dict(annotations=[], modulations=[], snrs=[], channels=[], sample_rate=150e3,
                                 sample_num=1200)
        validation_annotations = dict(annotations=[], modulations=[], snrs=[], channels=[], sample_rate=150e3,
                                      sample_num=1200)

        snrs = []
        channels = []
        modulations = []
        for item_index, item_indice in enumerate(tqdm(item_indices)):
            anno = IoLoad(jsons[item_indice])
            if item_indice in train_indices:
                train_annotations['annotations'].append(anno)
            elif item_indice in test_indices:
                test_annotations['annotations'].append(anno)
            else:
                validation_annotations['annotations'].append(anno)
            snrs.extend(anno['snr'])
            channels.extend(anno['channel'])
            modulations.extend(anno['modulation'])

        snrs = sorted(list(set(snrs)))
        channels = sorted(list(set(channels)))
        modulations = sorted(list(set(modulations)))

        train_annotations['snrs'] = snrs
        train_annotations['channels'] = channels
        train_annotations['modulations'] = modulations

        test_annotations['snrs'] = snrs
        test_annotations['channels'] = channels
        test_annotations['modulations'] = modulations

        validation_annotations['snrs'] = snrs
        validation_annotations['channels'] = channels
        validation_annotations['modulations'] = modulations

        return train_annotations, validation_annotations, test_annotations

    def generate(self):
        try:
            train_annotations, validation_annotations, test_annotations = self.preprocess_original_data

            train_and_validation_annotations = combine_two_annotations(train_annotations, validation_annotations)

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
