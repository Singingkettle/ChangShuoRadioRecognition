from ..builder import EVALUATES, MERGES
from ..utils import get_classification_accuracy_for_evaluation, get_online_classification_accuracy_for_evaluation, reshape_results
from ...common.utils import build_from_cfg


@EVALUATES.register_module()
class EvaluateSingleModulationPrediction:
    def __init__(self, prediction_name):
        self.prediction_name = prediction_name

    def __call__(self, results, data_infos):
        snr_to_index = data_infos['snr_to_index']
        item_snr_index = data_infos['item_snr_index']
        snr_num = len(snr_to_index)
        mod_label_num = len(data_infos['mod_to_label'])
        item_mod_label = data_infos['item_mod_label']
        results = reshape_results(results[self.prediction_name], mod_label_num)
        eval_results = get_classification_accuracy_for_evaluation(snr_num, mod_label_num, snr_to_index, item_snr_index,
                                                                  results, item_mod_label,
                                                                  prefix=self.prediction_name + '/')

        return eval_results


@EVALUATES.register_module()
class EvaluateMultiModulationPrediction:
    def __init__(self, prediction_names, merge=None):
        self.prediction_names = prediction_names
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
        for pr_name in self.prediction_names[1:]:
            selected_results[pr_name] = results[pr_name]

        if self.merge:
            selected_results = self.merge(selected_results, data_infos, self.prediction_names[0])

        for pr_name in selected_results:
            sub_results = reshape_results(selected_results[pr_name], mod_label_num)
            sub_eval_results = get_classification_accuracy_for_evaluation(snr_num, mod_label_num, snr_to_index,
                                                                          item_snr_index, sub_results, item_mod_label,
                                                                          prefix=pr_name + '/')
            eval_results.update(sub_eval_results)

        return eval_results


@EVALUATES.register_module()
class EvaluateSNRPrediction:
    def __init__(self, prediction_name='snr'):
        self.prediction_name = prediction_name

    def __call__(self, results, data_infos):
        snr_to_index = data_infos['snr_to_index']
        item_snr_index = data_infos['item_snr_index']
        snr_num = len(snr_to_index)
        snr_label_num = len(data_infos['snr_to_label'])
        item_snr_label = data_infos['item_snr_label']
        results = reshape_results(results[self.prediction_name], snr_label_num)
        eval_results = get_classification_accuracy_for_evaluation(snr_num, snr_label_num, snr_to_index, item_snr_index,
                                                                  results, item_snr_label,
                                                                  prefix=self.prediction_name + '/')

        return eval_results


@EVALUATES.register_module()
class EvaluateOnlineSingleModulationPrediction:
    def __init__(self, prediction_name):
        self.prediction_name = prediction_name

    def __call__(self, results, data_infos):
        mod_label_num = len(data_infos['mod_to_label'])
        item_mod_label = data_infos['item_mod_label']
        results = reshape_results(results[self.prediction_name], mod_label_num)
        eval_results = get_online_classification_accuracy_for_evaluation(mod_label_num, results, item_mod_label,
                                                                         prefix=self.prediction_name + '/')

        return eval_results