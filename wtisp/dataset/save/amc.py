import os.path as osp
import pickle

from ..builder import SAVES, MERGES
from ..utils import reshape_results, get_confusion_matrix
from ...common.utils import build_from_cfg


@SAVES.register_module()
class SaveSingleModulationPrediction:
    def __init__(self, prediction_name='final'):
        self.prediction_name = prediction_name

    def __call__(self, out_dir, results, data_infos, CLASSES, SNRS):
        snr_to_index = data_infos['snr_to_index']
        item_snr_index = data_infos['item_snr_index']
        snr_num = len(snr_to_index)
        mod_label_num = len(data_infos['mod_to_label'])
        item_mod_label = data_infos['item_mod_label']
        results = reshape_results(results[self.prediction_name], mod_label_num)
        confusion_matrix = get_confusion_matrix(snr_num, mod_label_num, item_snr_index, results, item_mod_label)
        save_res = dict(pre=results, cm=confusion_matrix, cl=CLASSES, sn=SNRS)
        save_path = osp.join(out_dir, self.prediction_name + '.pkl')
        pickle.dump(save_res, open(save_path, 'wb'))


@SAVES.register_module()
class SaveMultiModulationPrediction:
    def __init__(self, prediction_names=None, merge=None):
        self.prediction_names = prediction_names
        if merge:
            self.merge = build_from_cfg(merge, MERGES)
        else:
            self.merge = None

    def __call__(self, out_dir, results, data_infos, CLASSES, SNRS):
        snr_to_index = data_infos['snr_to_index']
        item_snr_index = data_infos['item_snr_index']
        snr_num = len(snr_to_index)
        mod_label_num = len(data_infos['mod_to_label'])
        item_mod_label = data_infos['item_mod_label']

        selected_results = dict()
        for pr_name in self.prediction_names:
            selected_results[pr_name] = results[pr_name]

        if self.merge:
            selected_results = self.merge.merge(selected_results, data_infos)

        for pr_name in selected_results:
            sub_results = reshape_results(selected_results[pr_name], mod_label_num)
            confusion_matrix = get_confusion_matrix(snr_num, mod_label_num, item_snr_index, sub_results, item_mod_label)
            save_res = dict(pre=sub_results, cm=confusion_matrix, cl=CLASSES, sn=SNRS)
            save_path = osp.join(out_dir, pr_name + '.pkl')
            pickle.dump(save_res, open(save_path, 'wb'))


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
        results = reshape_results(results[self.prediction_name], snr_label_num)
        confusion_matrix = get_confusion_matrix(snr_num, snr_label_num, item_snr_index, results, item_snr_label)
        save_res = dict(pre=results, cm=confusion_matrix, cl=SNRS, sn=SNRS)
        save_path = osp.join(out_dir, self.prediction_name + '.pkl')
        pickle.dump(save_res, open(save_path, 'wb'))