from typing import Dict, Any, List

import numpy as np

from ..builder import EVALUATES
from csrr.performance.metrics import ClassificationMetricsWithSNR


@EVALUATES.register_module()
class EvaluateSingleHeadClassifierWithSNR:

    def __init__(self, target_name: str, metrics: str = None):
        if metrics is None:
            metrics = ['accuracy']
        self.target_name = target_name
        self.metrics = metrics

    def __call__(self, results: List[np.ndarray], data_infos: Dict[str, Any]) -> Dict[str, Any]:
        gts = []
        snrs = []
        for annotation in data_infos['annotations']:
            gt = data_infos['modulations'].index(annotation[self.target_name])
            gts.append(gt)
            snrs.append(annotation['snr'])
        gts = np.array(gts, dtype=np.float64)
        pps = np.stack(results, axis=0)
        snrs = np.array(snrs, dtype=np.int64)
        performance_generator = ClassificationMetricsWithSNR(pps, gts, snrs)
        eval_results = dict()
        for metric in self.metrics:
            eval_results.update(getattr(performance_generator, metric))
        return eval_results

# @EVALUATES.register_module()
# class EvaluateClassificationWithSNROfHCGDNN:
#     def __init__(self, merge=None):
#         if merge is not None:
#             self.merge = build_from_cfg(merge, MERGES)
#         else:
#             self.merge = None
#             raise ValueError('You should give a fusion strategy for HCGDNN model!')
#
#     def __call__(self, results, data_infos):
#         results = self.merge(results, data_infos, 'Final')
#         eval_results = generate_method_eval_results(results, data_infos, 'mod')
#         return eval_results
#
#
# @EVALUATES.register_module()
# class EvaluateSNRPrediction:
#     def __init__(self, prediction_name=None):
#         self.prediction_name = prediction_name
#
#     def __call__(self, results, data_infos):
#
#         snr_to_index = data_infos['snr_to_index']
#         item_snr_index = data_infos['item_snr_index']
#         num_snr = len(snr_to_index)
#         snr_label_num = len(data_infos['snr_to_label'])
#         item_snr_label = data_infos['item_snr_label']
#
#         if self.prediction_name is None:
#             for pr_name in results:
#                 if 'SNR' in pr_name:
#                     self.prediction_name = pr_name
#             if self.prediction_name is None:
#                 raise ValueError('You should check your method code to make sure there is a group of SNR prediction!')
#         results = reshape_results(results[self.prediction_name], snr_label_num)
#         eval_results = get_classification_accuracy_with_snr(num_snr, snr_label_num, snr_to_index, item_snr_index,
#                                                             results, item_snr_label,
#                                                             prefix=self.prediction_name + '/')
#
#         return eval_results
#
#
# @EVALUATES.register_module()
# class EvaluateOnlineModulationPrediction:
#     def __init__(self, prediction_name=None):
#         self.prediction_name = prediction_name
#
#     def __call__(self, results, data_infos):
#         mod_label_num = len(data_infos['mod_to_label'])
#         item_mod_label = data_infos['item_mod_label']
#         selected_results = dict()
#         eval_results = dict()
#
#         for pr_name in results:
#             sub_results = reshape_results(selected_results[pr_name], mod_label_num)
#             sub_eval_results = get_online_classification_accuracy_for_evaluation(mod_label_num, sub_results,
#                                                                                  item_mod_label,
#                                                                                  prefix=pr_name + '/')
#             eval_results.update(sub_eval_results)
#
#         return eval_results
#
#
# @EVALUATES.register_module()
# class EvaluateTwoClassificationTaskWithSNR:
#     def __init__(self, method1='AMC', method2='SEI'):
#         self.method1 = method1
#         self.method2 = method2
#         self.method1_eval = EvaluateClassificationWithSNR(method=method1)
#         self.method2_eval = EvaluateClassificationWithSNR(method=method2)
#
#     def __call__(self, results, data_infos):
#         eval_results = dict()
#         for out_name in results:
#             sub_eval_results = None
#             if self.method1 in out_name:
#                 sub_eval_results = self.method1_eval({out_name: results[out_name]}, data_infos)
#             if self.method2 in out_name:
#                 sub_eval_results = self.method2_eval({out_name: results[out_name]}, data_infos)
#             if sub_eval_results:
#                 eval_results.update(sub_eval_results)
#         return eval_results
