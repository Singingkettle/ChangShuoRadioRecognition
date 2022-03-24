from ..builder import EVALUATES, MERGES
from ..utils import get_classification_accuracy_for_evaluation, get_online_classification_accuracy_for_evaluation, \
    reshape_results
from ...common.utils import build_from_cfg


@EVALUATES.register_module()
class EvaluateModulationPrediction:
    def __init__(self, prediction_name=None, merge=None):
        self.prediction_name = prediction_name
        if merge:
            self.merge = build_from_cfg(merge, MERGES)
        else:
            self.merge = None

    def __call__(self, results, data_infos):
        snr_to_index = data_infos['snr_to_index']
        item_snr_index = data_infos['item_snr_index']
        snr_num = len(snr_to_index)
        mod_label_num = len(data_infos['mod_to_label'])
        item_mod_label = data_infos['item_mod_label']

        selected_results = dict()
        eval_results = dict()

        if self.prediction_name is None:
            final_method_name = None
            is_not_merge = True
            for pr_name in results:
                if results[pr_name][0] is None:
                    final_method_name = pr_name
                    results.pop(pr_name, None)
                    is_not_merge = False
                    break
            selected_results = results
            if is_not_merge:
                self.merge = None
        else:
            for pr_name in self.prediction_name[1:]:
                selected_results[pr_name] = results[pr_name]
            final_method_name = self.prediction_name[0]

        if self.merge:
            selected_results = self.merge(selected_results, data_infos, final_method_name)

        for pr_name in selected_results:
            sub_results = reshape_results(selected_results[pr_name], mod_label_num)
            sub_eval_results = get_classification_accuracy_for_evaluation(snr_num, mod_label_num, snr_to_index,
                                                                          item_snr_index, sub_results, item_mod_label,
                                                                          prefix=pr_name + '/')
            eval_results.update(sub_eval_results)

        return eval_results


@EVALUATES.register_module()
class EvaluateSNRPrediction:
    def __init__(self, prediction_name=None):
        self.prediction_name = prediction_name

    def __call__(self, results, data_infos):

        snr_to_index = data_infos['snr_to_index']
        item_snr_index = data_infos['item_snr_index']
        snr_num = len(snr_to_index)
        snr_label_num = len(data_infos['snr_to_label'])
        item_snr_label = data_infos['item_snr_label']

        if self.prediction_name is None:
            for pr_name in results:
                if 'SNR' in pr_name:
                    self.prediction_name = pr_name
            if self.prediction_name is None:
                raise ValueError('You should check your task code to make sure there is a group of SNR prediction!')
        results = reshape_results(results[self.prediction_name], snr_label_num)
        eval_results = get_classification_accuracy_for_evaluation(snr_num, snr_label_num, snr_to_index, item_snr_index,
                                                                  results, item_snr_label,
                                                                  prefix=self.prediction_name + '/')

        return eval_results


@EVALUATES.register_module()
class EvaluateOnlineModulationPrediction:
    def __init__(self, prediction_name=None, merge=None):
        self.prediction_name = prediction_name
        if merge:
            self.merge = build_from_cfg(merge, MERGES)
        else:
            self.merge = None

    def __call__(self, results, data_infos):
        mod_label_num = len(data_infos['mod_to_label'])
        item_mod_label = data_infos['item_mod_label']
        selected_results = dict()
        eval_results = dict()

        if self.prediction_name is None:
            final_method_name = None
            is_not_merge = True
            for pr_name in results:
                if results[pr_name][0] is None:
                    final_method_name = pr_name
                    results.pop(pr_name, None)
                    is_not_merge = False
                    break
            selected_results = results
            if is_not_merge:
                self.merge = None
        else:
            for pr_name in self.prediction_name[1:]:
                selected_results[pr_name] = results[pr_name]
            final_method_name = self.prediction_name[0]

        if self.merge:
            selected_results = self.merge(selected_results, data_infos, final_method_name)

        for pr_name in selected_results:
            sub_results = reshape_results(selected_results[pr_name], mod_label_num)
            sub_eval_results = get_online_classification_accuracy_for_evaluation(mod_label_num, sub_results,
                                                                                 item_mod_label,
                                                                                 prefix=self.prediction_name + '/')
            eval_results.update(sub_eval_results)

        return eval_results
