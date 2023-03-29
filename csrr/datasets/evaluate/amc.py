from typing import Dict, Any, List

import numpy as np

from ..builder import EVALUATES, MERGES, build_from_cfg
from ..utils import list_dict_to_dict_list
from ...performance.metrics import get_classification_eval_with_snr


@EVALUATES.register_module()
class EvaluateSingleHeadClassifierWithSNR:

    def __init__(self, target_name: str, metrics: str = None):
        if metrics is None:
            metrics = ['ACC']
        self.target_name = target_name
        self.metrics = metrics

    def __call__(self, results: List[np.ndarray], data_infos: Dict[str, Any]) -> Dict[str, Any]:
        gts = []
        snrs = []
        for annotation in data_infos['annotations']:
            gt = data_infos[f'{self.target_name}s'].index(annotation[self.target_name])
            gts.append(gt)
            snrs.append(annotation['snr'])

        gts = np.array(gts, dtype=np.float64)
        pps = np.stack(results, axis=0)
        snrs = np.array(snrs, dtype=np.int64)
        eval_results = get_classification_eval_with_snr(pps, gts, snrs, data_infos[f'{self.target_name}s'],
                                                        self.metrics)
        return eval_results


@EVALUATES.register_module()
class EvaluateMLDNN:

    def __init__(self, snr_threshold=0, metrics: str = None):
        self.snr_threshold = snr_threshold
        if metrics is None:
            metrics = ['ACC']
        self.metrics = metrics

    def __call__(self, results: List[Dict[str, np.ndarray]], data_infos: Dict[str, Any]) -> Dict[str, Any]:
        gts = []
        snr_gts = []
        snrs = []
        for annotation in data_infos['annotations']:
            gt = data_infos['modulations'].index(annotation['modulation'])
            if annotation['snr'] >= self.snr_threshold:
                snr_gt = 0
            else:
                snr_gt = 1
            gts.append(gt)
            snr_gts.append(snr_gt)
            snrs.append(annotation['snr'])
        results = list_dict_to_dict_list(results)
        gts = np.array(gts, dtype=np.float64)
        snr_gts = np.array(snr_gts, dtype=np.float64)
        eval_results = dict()
        for key in results:
            pps = np.stack(results[key], axis=0)
            if key == 'snr':
                res = get_classification_eval_with_snr(pps, snr_gts, snrs, ['HighSNR', 'LowSNR'], self.metrics)
            else:
                res = get_classification_eval_with_snr(pps, gts, snrs, data_infos['modulations'], self.metrics)
            if key == 'merge':
                eval_results.update(res)
            else:
                eval_results.update({f'{k}_{key}': v for k, v in res.items()})

        return eval_results


@EVALUATES.register_module()
class EvaluateHCGDNN:
    def __init__(self, metrics: str = None, merge: Dict[str, str] = None):
        if metrics is None:
            metrics = ['ACC']
        self.metrics = metrics
        if merge is not None:
            self.merge = build_from_cfg(merge, MERGES)
        else:
            self.merge = None
            raise ValueError('You should give a fusion strategy for HCGDNN model!')

    def __call__(self, results, data_infos):
        gts = []
        snrs = []
        eval_results = dict()
        for annotation in data_infos['annotations']:
            gt = data_infos['modulations'].index(annotation['modulation'])
            gts.append(gt)
            snrs.append(annotation['snr'])
        results = list_dict_to_dict_list(results)
        gts = np.array(gts, dtype=np.float64)
        snrs = np.array(snrs, dtype=np.int64)

        mpps = []
        for key in results:
            pps = np.stack(results[key], axis=0)
            res = get_classification_eval_with_snr(pps, gts, snrs, data_infos['modulations'], self.metrics)
            eval_results.update({f'{k}_{key}': v for k, v in res.items()})
            mpps.append(pps)
        pps = self.merge(mpps, gts)
        res = get_classification_eval_with_snr(pps, gts, snrs, data_infos['modulations'], self.metrics)
        eval_results.update(res)
        return eval_results


@EVALUATES.register_module()
class EvaluateFastMLDNN:
    def __init__(self, target_name: str, metrics: str = None):
        if metrics is None:
            metrics = ['ACC', 'FeaDistribution']
        self.target_name = target_name
        self.metrics = metrics

    def __call__(self, results, data_infos):
        gts = []
        snrs = []
        for annotation in data_infos['annotations']:
            gt = data_infos[f'{self.target_name}s'].index(annotation[self.target_name])
            gts.append(gt)
            snrs.append(annotation['snr'])
        results = list_dict_to_dict_list(results)

        gts = np.array(gts, dtype=np.float64)
        pps = np.stack(results['pre'], axis=0)
        snrs = np.array(snrs, dtype=np.int64)
        eval_results = get_classification_eval_with_snr(pps, gts, snrs, data_infos[f'{self.target_name}s'],
                                                        self.metrics, feas=np.stack(results['fea'], axis=0),
                                                        centers=results['center'][0])

        return eval_results

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
