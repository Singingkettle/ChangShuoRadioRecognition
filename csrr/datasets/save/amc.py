import copy
import os.path as osp
import pickle

from ..builder import SAVES, MERGES
from ..utils import reshape_results, get_confusion_matrix
from ...common.utils import build_from_cfg


def generate_amc_save_pkl(out_dir, results, data_infos, CLASSES, SNRS):
    snr_to_index = data_infos['snr_to_index']
    item_snr_index = data_infos['item_snr_index']
    snr_num = len(snr_to_index)
    mod_label_num = len(data_infos['mod_to_label'])
    item_mod_label = data_infos['item_mod_label']

    for pr_name in results:
        if 'SNR' in pr_name:
            continue
        sub_results = reshape_results(results[pr_name], mod_label_num)
        confusion_matrix = get_confusion_matrix(snr_num, mod_label_num, item_snr_index, sub_results, item_mod_label)
        save_res = dict(probabilities=sub_results, confusion_mnatrix=confusion_matrix, CLASSES=CLASSES, SNRS=SNRS)
        save_path = osp.join(out_dir, pr_name + '.pkl')
        pickle.dump(save_res, open(save_path, 'wb'))


@SAVES.register_module()
class SaveModulationPrediction:
    def __init__(self, prediction_names=None):
        self.prediction_names = prediction_names

    def __call__(self, out_dir, results, data_infos, CLASSES, SNRS):
        if self.prediction_names is not None:
            selected_results = dict()
            for pr_name in results:
                selected_results[pr_name] = copy.deepcopy(results[pr_name])
        else:
            selected_results = results
        generate_amc_save_pkl(out_dir, selected_results, data_infos, CLASSES, SNRS)


@SAVES.register_module()
class SaveModulationPredictionOfHCGDNN:
    def __init__(self, merge=None):
        if merge is not None:
            self.merge = build_from_cfg(merge, MERGES)
        else:
            self.merge = None
            raise ValueError('You should give a fusion strategy for HCGDNN model!')

    def __call__(self, out_dir, results, data_infos, CLASSES, SNRS):
        results = self.merge(results, data_infos, self.method_name)
        generate_amc_save_pkl(out_dir, results, data_infos, CLASSES, SNRS)


@SAVES.register_module()
class SaveSNRPrediction:
    def __init__(self, prediction_name='snr'):
        self.prediction_name = prediction_name

    def __call__(self, out_dir, results, data_infos, CLASSES, SNRS):
        snr_to_index = data_infos['snr_to_index']
        item_snr_index = data_infos['item_snr_index']
        snr_num = len(snr_to_index)
        snr_label_num = len(data_infos['snr_label'])
        item_snr_label = data_infos['item_snr_label']

        if self.prediction_name is None:
            for pr_name in results:
                if 'SNR' in pr_name:
                    self.prediction_name = pr_name
            if self.prediction_name is None:
                raise ValueError('You should check your task code to make sure there is a group of SNR prediction!')

        results = reshape_results(results[self.prediction_name], snr_label_num)
        confusion_matrix = get_confusion_matrix(snr_num, snr_label_num, item_snr_index, results, item_snr_label)
        save_res = dict(probabilities=results, confusion_mnatrix=confusion_matrix, CLASSES=CLASSES, SNRS=SNRS)
        save_path = osp.join(out_dir, self.prediction_name + '.pkl')
        pickle.dump(save_res, open(save_path, 'wb'))
