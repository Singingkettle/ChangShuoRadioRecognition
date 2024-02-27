import logging
from copy import copy
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from mmengine.dist import (broadcast_object_list, collect_results,
                           is_main_process)
from mmengine.evaluator import BaseMetric
from mmengine.evaluator.metric import _to_cpu
from mmengine.logging import print_log
from scipy.optimize import minimize, Bounds, LinearConstraint
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize

from csrr.registry import METRICS
from .single_label import to_tensor


def _by_gridsearch(mpps, gts, grid_step, outputs):
    mpps = np.stack(mpps, axis=0)
    m, n, c = mpps.shape

    def get_merge_weight_by_grid_search(num_pre, search_step):
        def search(search_depth, cur_sum):
            if search_depth == 1:
                return [[cur_sum]]
            search_depth -= 1
            cur_val = 0
            cur_res = []
            while cur_val <= cur_sum:
                sub_res = search(search_depth, cur_sum - cur_val)
                for sample in sub_res:
                    new_sample = copy.deepcopy(sample)
                    new_sample.append(cur_val)
                    cur_res.append(new_sample)
                cur_val += search_step

            return cur_res

        res = search(num_pre, 1)

        return res

    ws = get_merge_weight_by_grid_search(m, grid_step)
    ws = np.array(ws, dtype=np.float64)
    pps = np.reshape(mpps, (m, -1))
    pps = ws @ pps
    pps = np.reshape(pps, (-1, n, c))
    pps_ = np.argmax(pps, axis=2)

    max_acc = 0
    w = ws[0, :]
    best_pps = pps[0, :, :]
    for w_index in range(ws.shape[0]):
        cur_acc = accuracy_score(gts, pps_[w_index, :])
        if cur_acc > max_acc:
            max_acc = cur_acc
            w = ws[w_index, :]
            best_pps = pps[w_index, :, :]

    metrics = {head_name: w[i] for i, head_name in enumerate(outputs)}

    return metrics, best_pps


def _by_optimization(mpps, gts, method, outputs):
    mpps = np.stack(mpps, axis=0)
    m, n, c = mpps.shape
    gts = label_binarize(gts, classes=[i for i in range(c)])

    pre_max_index = np.argmax(mpps, axis=2)
    gt_max_index = np.argmax(gts, axis=1)
    diff_index = pre_max_index - np.reshape(gt_max_index, (1, -1))
    no_zero_index = np.nonzero((np.sum(np.abs(diff_index), axis=0)))[0]

    bad_pre_matrix = mpps[:, no_zero_index[:], :]
    targets = gts[no_zero_index[:], :]

    def get_merge_weight_by_optimization(x, t, method):
        x = x.astype(dtype=np.float64)
        t = t.astype(dtype=np.float64)
        m = x.shape[0]
        n = x.shape[1]
        tau = 1000 / (np.max(x[:]) + np.finfo(np.float64).eps)

        r1 = 2 * tau / n
        r2 = 2 * tau * tau / n

        def min_obj(w):
            w = np.reshape(w, (-1, 1, 1))
            y0 = x * w
            y1 = np.sum(y0, axis=0)
            y2 = y1 * tau
            y2 = y2 - np.max(y2, axis=1)[:, None]
            y3 = np.exp(y2)
            y4 = np.sum(y3, axis=1)[:, None]
            y5 = y3 / y4

            y6 = (y5 - t)
            y7 = np.power(y6, 2)
            f = np.mean(y7[:])
            return f

        def obj_der(w):
            w = np.reshape(w, (-1, 1, 1))
            y0 = x * w
            y1 = np.sum(y0, axis=0)
            y2 = y1 * tau
            y2 = y2 - np.max(y2, axis=1)[:, None]
            y3 = np.exp(y2)
            y4 = np.sum(y3, axis=1)[:, None]
            y5 = y3 / y4

            y6 = y5 * (y5 - t)[None, :, :]

            y7 = y5[None, :, :]
            y7 = y7 * x
            y7 = np.sum(y7, axis=2)[:, :, None]

            y8 = y6 * (x - y7)

            df = np.sum(np.sum(y8, axis=2), axis=1) * r1

            return df

        def obj_hess(w):
            w = np.reshape(w, (-1, 1, 1))
            y0 = x * w
            y1 = np.sum(y0, axis=0)
            y2 = y1 * tau
            y2 = y2 - np.max(y2, axis=1)[:, None]
            y3 = np.exp(y2)
            y4 = np.sum(y3, axis=1)[:, None]
            y5 = y3 / y4
            y6 = y5[None, :, :]

            y7_ = y5 * (2 * y5 - t)
            y7 = y7_[None, :, :]
            y8_ = np.sum(y6 * x, axis=2)
            y8 = y8_[:, :, None]
            y9 = x - y8
            y10 = y7 * y9

            y11 = np.reshape(y9, (m, -1))
            y12 = np.reshape(y10, (m, -1))
            H1 = y11 @ y12.T

            y13 = y5 * (y5 - t)
            y13 = np.sum(y13, axis=1)
            y13 = np.reshape(y13, (1, -1))

            y14 = y8_ * y13
            y15 = y14 @ y8_.T

            y16 = y6 * x * y13[:, :, None]
            y17 = np.reshape(y16, (m, -1))
            y18 = np.reshape(x, (m, -1))

            H2 = y15 - y17 @ y18.T

            H = H1 + H2

            H = H * r2

            return H

        w0 = np.ones((m,), dtype=np.float64)/m
        w0[-1] = 1 - np.sum(w0[:-1])

        lb = [0, ] * m
        ub = [1, ] * m
        bounds = Bounds(lb, ub)

        A = [[1, ] * m, ]
        lb = [1, ]
        ub = [1, ]
        linear_constraint = LinearConstraint(A, lb, ub)
        res = minimize(min_obj, w0, method=method, jac=obj_der, hess=obj_hess,
                       constraints=[linear_constraint, ],
                       options={'verbose': 1}, bounds=bounds)
        best_w = res.x
        return best_w

    w = get_merge_weight_by_optimization(bad_pre_matrix, targets, method)
    mpps = np.dot(w.T, np.reshape(mpps, (m, -1)))
    best_pps = np.reshape(mpps, (n, c))

    metrics = {head_name: w[i] for i, head_name in enumerate(outputs)}

    return metrics, best_pps


@METRICS.register_module()
class HCGDNNWeightsAccuracy(BaseMetric):
    r"""Weights for fusion, and Accuracy evaluation metric.
    """
    default_prefix: Optional[str] = 'accuracy'

    def __init__(self,
                 topk: Union[int, Sequence[int]] = (1,),
                 thrs: Union[float, Sequence[Union[float, None]], None] = 0.,
                 collect_device: str = 'cpu',
                 weights: Dict[str, Union[str, float]] = dict(optimization='trust-constr'),
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        if isinstance(topk, int):
            self.topk = (topk,)
        else:
            self.topk = tuple(topk)

        if isinstance(thrs, float) or thrs is None:
            self.thrs = (thrs,)
        else:
            self.thrs = tuple(thrs)

        self.weights = weights

    def process(self, data_batch, data_samples: Sequence[dict]):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        for data_sample in data_samples:
            result = dict()
            if 'cnn_pred_score' in data_sample:
                result['cnn_pred_score'] = data_sample['cnn_pred_score'].cpu()
            if 'gru1_pred_score' in data_sample:
                result['gru1_pred_score'] = data_sample['gru1_pred_score'].cpu()
            if 'gru2_pred_score' in data_sample:
                result['gru2_pred_score'] = data_sample['gru2_pred_score'].cpu()
            result['gt_label'] = data_sample['gt_label'].cpu()
            # Save the result to `self.results`.
            self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # NOTICE: don't access `self.results` from the method.
        metrics = {}

        # calculate fusion weights
        mpps = []
        outputs = []
        results = self.results
        results = {k: [dic[k] for dic in results] for k in results[0]}
        scores = dict()
        if 'cnn_pred_score' in results:
            cnn_scores = torch.stack(results['cnn_pred_score'])
            scores['cnn'] = cnn_scores
            mpps.append(cnn_scores.cpu().numpy())
            outputs.append('cnn')
        if 'gru1_pred_score' in results:
            gru1_scores = torch.stack(results['gru1_pred_score'])
            scores['gru1'] = gru1_scores
            mpps.append(gru1_scores.cpu().numpy())
            outputs.append('gru1')
        if 'gru2_pred_score' in results:
            gru2_scores = torch.stack(results['gru2_pred_score'])
            scores['gru2'] = gru2_scores
            mpps.append(gru2_scores.cpu().numpy())
            outputs.append('gru2')

        # concat
        target = torch.cat(results['gt_label'])

        gts = target.cpu().numpy()
        if 'optimization' in self.weights:
            sub_metrics, pred = _by_optimization(mpps, gts, self.weights['optimization'], outputs)
        elif 'gridsearch' in self.weights:
            sub_metrics, pred = _by_gridsearch(mpps, gts, self.weights['gridsearch'], outputs)
        else:
            raise NotImplementedError(
                'Currently, only support optimization and gridsearch to solve the fusion weights in HCGDNN!')

        scores['merge'] = pred

        def _accuracy_metrics(pred, target, prefix=''):
            _metrics = dict()
            try:
                acc = self.calculate(pred, target, self.topk, self.thrs)
            except ValueError as e:
                # If the topk is invalid.
                raise ValueError(
                    str(e) + ' Please check the `val_evaluator` and '
                             '`test_evaluator` fields in your config file.')

            multi_thrs = len(self.thrs) > 1
            for i, k in enumerate(self.topk):
                for j, thr in enumerate(self.thrs):
                    name = f'{prefix}top{k}'
                    if multi_thrs:
                        name += '_no-thr' if thr is None else f'_thr-{thr:.2f}'
                    _metrics[name] = acc[i][j].item()

            return _metrics

        metrics.update(sub_metrics)
        for head_name in scores:
            if 'merge' == head_name:
                acc_metrics = _accuracy_metrics(scores[head_name], target)
            else:
                acc_metrics = _accuracy_metrics(scores[head_name], target, prefix=head_name)
            metrics.update(acc_metrics)

        return metrics

    @staticmethod
    def calculate(
            pred: Union[torch.Tensor, np.ndarray, Sequence],
            target: Union[torch.Tensor, np.ndarray, Sequence],
            topk: Sequence[int] = (1,),
            thrs: Sequence[Union[float, None]] = (0.,),
    ) -> Union[torch.Tensor, List[List[torch.Tensor]]]:
        """Calculate the accuracy.

        Args:
            pred (torch.Tensor | np.ndarray | Sequence): The prediction
                results. It can be labels (N, ), or scores of every
                class (N, C).
            target (torch.Tensor | np.ndarray | Sequence): The target of
                each prediction with shape (N, ).
            thrs (Sequence[float | None]): Predictions with scores under
                the thresholds are considered negative. It's only used
                when ``pred`` is scores. None means no thresholds.
                Defaults to (0., ).
            thrs (Sequence[float]): Predictions with scores under
                the thresholds are considered negative. It's only used
                when ``pred`` is scores. Defaults to (0., ).

        Returns:
            torch.Tensor | List[List[torch.Tensor]]: Accuracy.

            - torch.Tensor: If the ``pred`` is a sequence of label instead of
              score (number of dimensions is 1). Only return a top-1 accuracy
              tensor, and ignore the argument ``topk` and ``thrs``.
            - List[List[torch.Tensor]]: If the ``pred`` is a sequence of score
              (number of dimensions is 2). Return the accuracy on each ``topk``
              and ``thrs``. And the first dim is ``topk``, the second dim is
              ``thrs``.
        """

        pred = to_tensor(pred)
        target = to_tensor(target).to(torch.int64)
        num = pred.size(0)
        assert pred.size(0) == target.size(0), \
            f"The size of pred ({pred.size(0)}) doesn't match " \
            f'the target ({target.size(0)}).'

        if pred.ndim == 1:
            # For pred label, ignore topk and acc
            pred_label = pred.int()
            correct = pred.eq(target).float().sum(0, keepdim=True)
            acc = correct.mul_(100. / num)
            return acc
        else:
            # For pred score, calculate on all topk and thresholds.
            pred = pred.float()
            maxk = max(topk)

            if maxk > pred.size(1):
                raise ValueError(
                    f'Top-{maxk} accuracy is unavailable since the number of '
                    f'categories is {pred.size(1)}.')

            pred_score, pred_label = pred.topk(maxk, dim=1)
            pred_label = pred_label.t()
            correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
            results = []
            for k in topk:
                results.append([])
                for thr in thrs:
                    # Only prediction values larger than thr are counted
                    # as correct
                    _correct = correct
                    if thr is not None:
                        _correct = _correct & (pred_score.t() > thr)
                    correct_k = _correct[:k].reshape(-1).float().sum(
                        0, keepdim=True)
                    acc = correct_k.mul_(100. / num)
                    results[-1].append(acc)
            return results

    def evaluate(self, size: int) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.

        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
            names of the metrics, and the values are corresponding results.
        """
        if len(self.results) == 0:
            print_log(
                f'{self.__class__.__name__} got empty `self.results`. Please '
                'ensure that the processed results are properly added into '
                '`self.results` in `process` method.',
                logger='current',
                level=logging.WARNING)

        if self.collect_device == 'cpu':
            results = collect_results(
                self.results,
                size,
                self.collect_device,
                tmpdir=self.collect_dir)
        else:
            results = collect_results(self.results, size, self.collect_device)

        if is_main_process():
            # cast all tensors in results list to cpu
            results = _to_cpu(results)
            _metrics = self.compute_metrics(results)  # type: ignore

            def _prefix(k):
                if k == 'cnn':
                    return 'weights'
                elif k == 'gru1':
                    return 'weights'
                elif k == 'gru2':
                    return 'weights'
                else:
                    return self.prefix

            # Add prefix to metric names
            if self.prefix:
                _metrics = {
                    '/'.join((_prefix(k), k)): v
                    for k, v in _metrics.items()
                }
            metrics = [_metrics]
        else:
            metrics = [None]  # type: ignore

        broadcast_object_list(metrics)

        # reset the results list
        self.results.clear()
        return metrics[0]
