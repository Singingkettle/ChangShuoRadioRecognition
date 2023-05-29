import copy
import os
import pickle

import numpy as np

from .builder import DATASETS
from .custom import CustomDataset
from .utils import list_dict_to_dict_list
from ..performance.metrics import get_classification_eval_with_snr

@DATASETS.register_module()
class DeepSigDataset(CustomDataset):
    """Custom dataset for modulation classification.
    Args:
        ann_file (str): Annotation file path.
        data_root (str, optional): Data root for ``ann_file``, ``In-phase/Quadrature data``, ``Amplitude/Phase data``,
                                   ``constellation data``.
        test_mode (bool, optional): If set True, annotation will not be loaded.
    """
    CLASSES = None
    SNRS = None

    def __init__(self, ann_file, pipeline, data_root=None, test_mode=False,
                 preprocess=None, evaluate=None, format=None):
        super(DeepSigDataset, self).__init__(ann_file, pipeline, data_root, test_mode, preprocess, evaluate, format)

        self.CLASSES = self.data_infos['modulations']
        self.SNRS = self.data_infos['snrs']

    def get_ann_info(self, idx):
        results = dict()
        results['snr'] = self.data_infos['annotations'][idx]['snr']
        results['file_name'] = self.data_infos['annotations'][idx]['file_name']
        results['modulation'] = self.CLASSES.index(self.data_infos['annotations'][idx]['modulation'])
        results['snrs'] = self.SNRS
        results['modulations'] = self.CLASSES

        return results

    def pre_pipeline(self, results):
        results['data_root'] = self.data_root
        results['iq_folder'] = 'sequence_data/iq'
        results['ap_folder'] = 'sequence_data/ap'
        results['co_folder'] = 'constellation_data'
        results['inputs'] = dict()
        return results

    def paper(self, save_dir, results, cfg):
        gts = []
        snrs = []
        for annotation in self.data_infos['annotations']:
            gt = self.data_infos['modulations'].index(annotation['modulation'])
            gts.append(gt)
            snrs.append(annotation['snr'])
        gts = np.array(gts, dtype=np.float64)
        snrs = np.array(snrs, dtype=np.int64)
        if isinstance(results[0], dict):
            results = list_dict_to_dict_list(results)
            pps = np.stack(copy.deepcopy(results['pre']), axis=0)
            feas = np.stack(copy.deepcopy(results['fea']), axis=0)
            centers = copy.deepcopy(results['center'][0])
        else:
            pps = np.stack(results, axis=0)
            feas = None
            centers = None

        res = dict(gts=gts, pps=pps, snrs=snrs, classes=self.CLASSES, cfg=cfg, feas=feas, centers=centers)

        eval_results = get_classification_eval_with_snr(pps, gts, snrs, self.CLASSES, ['ACC'])
        print(eval_results)
        pickle.dump(res, open(os.path.join(save_dir, 'paper.pkl'), 'wb'), protocol=4)
