import copy

from ..builder import EVALUATES, MERGES
from ..utils import get_classification_accuracy_with_snr, get_online_classification_accuracy_for_evaluation, \
    reshape_results
from ...common.utils import build_from_cfg


def generate_task_eval_results(results, data_infos, class_name):
    eval_results = dict()
    snr_to_index = data_infos['snr_to_index']
    item_snr_index = data_infos['item_snr_index']
    snr_num = len(snr_to_index)
    class_label_num = len(data_infos[f'{class_name}_to_label'])
    item_class_label = data_infos[f'item_{class_name}_label']

    for pr_name in results:
        if 'SNR' in pr_name:
            continue
        if 'fea' in pr_name:
            continue
        sub_results = reshape_results(results[pr_name], class_label_num)
        sub_eval_results = get_classification_accuracy_with_snr(snr_num, class_label_num, snr_to_index,
                                                                item_snr_index, sub_results, item_class_label,
                                                                prefix=pr_name + '/')
        eval_results.update(sub_eval_results)

    return eval_results


@EVALUATES.register_module()
class EvaluateClassificationWithSNR:
    TASK = dict(AMC='mod', SEI='dev')

    def __init__(self, prediction_names=None, task='AMC'):
        self.prediction_names = prediction_names
        self.class_name = self.TASK[task]

    def __call__(self, results, data_infos):
        if self.prediction_names is not None:
            selected_results = dict()
            for pr_name in results:
                selected_results[pr_name] = copy.deepcopy(results[pr_name])
        else:
            selected_results = results
        eval_results = generate_task_eval_results(selected_results, data_infos, self.class_name)
        return eval_results


@EVALUATES.register_module()
class EvaluateClassificationWithSNROfHCGDNN:
    def __init__(self, merge=None):
        if merge is not None:
            self.merge = build_from_cfg(merge, MERGES)
        else:
            self.merge = None
            raise ValueError('You should give a fusion strategy for HCGDNN model!')

    def __call__(self, results, data_infos):
        results = self.merge(results, data_infos, 'Final')
        eval_results = generate_task_eval_results(results, data_infos, 'mod')
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
        eval_results = get_classification_accuracy_with_snr(snr_num, snr_label_num, snr_to_index, item_snr_index,
                                                            results, item_snr_label,
                                                            prefix=self.prediction_name + '/')

        return eval_results


@EVALUATES.register_module()
class EvaluateOnlineModulationPrediction:
    def __init__(self, prediction_name=None):
        self.prediction_name = prediction_name

    def __call__(self, results, data_infos):
        mod_label_num = len(data_infos['mod_to_label'])
        item_mod_label = data_infos['item_mod_label']
        selected_results = dict()
        eval_results = dict()

        for pr_name in results:
            sub_results = reshape_results(selected_results[pr_name], mod_label_num)
            sub_eval_results = get_online_classification_accuracy_for_evaluation(mod_label_num, sub_results,
                                                                                 item_mod_label,
                                                                                 prefix=pr_name + '/')
            eval_results.update(sub_eval_results)

        return eval_results


@EVALUATES.register_module()
class EvaluateTwoClassificationTaskWithSNR:
    def __init__(self, task1='AMC', task2='SEI'):
        self.task1 = task1
        self.task2 = task2
        self.task1_eval = EvaluateClassificationWithSNR(task=task1)
        self.task2_eval = EvaluateClassificationWithSNR(task=task2)

    def __call__(self, results, data_infos):
        eval_results = dict()
        for out_name in results:
            sub_eval_results = None
            if self.task1 in out_name:
                sub_eval_results = self.task1_eval({out_name: results[out_name]}, data_infos)
            if self.task2 in out_name:
                sub_eval_results = self.task2_eval({out_name: results[out_name]}, data_infos)
            if sub_eval_results:
                eval_results.update(sub_eval_results)
        return eval_results
