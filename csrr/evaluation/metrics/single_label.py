# Copyright (c) OpenMMLab. All rights reserved.
from itertools import combinations
from itertools import product
from typing import List, Optional, Sequence, Union

from sklearn.metrics import silhouette_score, silhouette_samples, roc_curve, auc

try:
    from tsnecuda import TSNE
except:
    from tsne_torch import TorchTSNE as TSNE

import mmengine
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.evaluator import BaseMetric

from csrr.registry import METRICS


def to_tensor(value):
    """Convert value to torch.Tensor."""
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    elif isinstance(value, Sequence) and not mmengine.is_str(value):
        value = torch.tensor(value)
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'{type(value)} is not an available argument.')
    return value


def to_numpy(value):
    """Convert value to np.ndarray."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    elif isinstance(value, Sequence) and not mmengine.is_str(value):
        value = np.asarray(value)
    elif not isinstance(value, np.ndarray):
        raise TypeError(f'{type(value)} is not an available argument.')
    return value


def _feature_distribution(feature, target, num_classes, vis_dim=2):
    feature = to_numpy(feature)
    silhouette_avg = silhouette_score(feature, target)
    sample_silhouette_values = silhouette_samples(feature, target)
    if vis_dim > 3:
        vis_dim = 3
    if feature.shape[1] > vis_dim:
        feature = TSNE(n_components=vis_dim, perplexity=15, learning_rate=10, verbose=1, n_jobs=12).fit_transform(
            feature)

    center = np.zeros((num_classes, feature.shape[1]), dtype=np.float32)
    for class_index in range(num_classes):
        center[class_index, :] = np.mean(feature[target == class_index, :], axis=0)

    return feature, center, silhouette_avg, sample_silhouette_values


def _roc_ovr_ovo(pred_positive, gt_positive, gt_label, num_classes):
    """This metric is based on https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    """
    pred_positive = to_numpy(pred_positive)
    gt_positive = to_numpy(gt_positive)
    gt_label = to_numpy(gt_label)

    def _roc_ovr(_pred_positive, _gt_positive, _num_classes):
        # In the first step, we calculate One-vs-Rest multiclass ROC in micro and macro
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(_num_classes):
            fpr[i], tpr[i], _ = roc_curve(_pred_positive[:, i], _gt_positive[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # ROC curve using micro-averaged OvR
        fpr['micro'], tpr['micro'], _ = roc_curve(_pred_positive.ravel(), _gt_positive.ravel())
        roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

        # ROC curve using the OvR macro-average
        fpr_grid = np.linspace(0.0, 1.0, 1000)

        # Interpolate all ROC curves at these points
        mean_tpr = np.zeros_like(fpr_grid)

        for i in range(_num_classes):
            mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

        # Average it and compute AUC
        mean_tpr /= _num_classes

        fpr['macro'] = fpr_grid
        tpr['macro'] = mean_tpr
        roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

        _roc = dict(fpr=fpr, tpr=tpr, auc=roc_auc)

        return _roc

    def _roc_ovo(_pred_positive, _gt_label, _num_classes):
        # In the second step, we calculate One-vs-One multiclass ROC in macro
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        fpr_grid = np.linspace(0.0, 1.0, 1000)
        pair_list = list(combinations([i for i in range(_num_classes)], 2))
        for ix, (label_a, label_b) in enumerate(pair_list):
            fpr[(label_a, label_b)] = dict()
            tpr[(label_a, label_b)] = dict()
            roc_auc[(label_a, label_b)] = dict()

            a_mask = _gt_label == label_a
            b_mask = _gt_label == label_b
            ab_mask = np.logical_or(a_mask, b_mask)

            a_true = a_mask[ab_mask]
            b_true = b_mask[ab_mask]

            fpr_a, tpr_a, _ = roc_curve(a_true, _pred_positive[ab_mask, label_a])
            fpr_b, tpr_b, _ = roc_curve(b_true, _pred_positive[ab_mask, label_b])

            auc_a = auc(fpr_a, tpr_a)
            auc_b = auc(fpr_b, tpr_b)

            fpr[(label_a, label_b)][label_a] = fpr_a
            fpr[(label_a, label_b)][label_b] = fpr_b
            tpr[(label_a, label_b)][label_a] = tpr_a
            tpr[(label_a, label_b)][label_b] = tpr_b
            roc_auc[(label_a, label_b)][label_a] = auc_a
            roc_auc[(label_a, label_b)][label_b] = auc_b

            mean_tpr = (np.interp(fpr_grid, fpr_a, tpr_a) + np.interp(fpr_grid, fpr_b, tpr_b)) / 2
            mean_auc = auc(fpr_grid, mean_tpr)

            fpr[(label_a, label_b)]['macro'] = fpr_grid
            tpr[(label_a, label_b)]['macro'] = mean_tpr
            roc_auc[(label_a, label_b)]['macro'] = mean_auc

        roc_auc['macro'] = sum([roc_auc[pair]['macro'] for pair in roc_auc.keys()]) / (1.0 * len(pair_list))

        _roc = dict(fpr=fpr, tpr=tpr, auc=roc_auc)

        return _roc

    # Save the result
    roc = dict()
    roc['ovr'] = _roc_ovr(pred_positive, gt_positive, num_classes)
    roc['ovo'] = _roc_ovo(pred_positive, gt_label, num_classes)

    return roc


def _precision_recall_f1_support(pred_positive, gt_positive, average):
    """calculate base classification task metrics, such as  precision, recall,
    f1_score, support."""
    average_options = ['micro', 'macro', None]
    assert average in average_options, 'Invalid `average` argument, ' \
                                       f'please specify from {average_options}.'

    # ignore -1 target such as difficult sample that is not wanted
    # in evaluation results.
    # only for calculate multi-label without affecting single-label behavior
    ignored_index = gt_positive == -1
    pred_positive[ignored_index] = 0
    gt_positive[ignored_index] = 0

    class_correct = (pred_positive & gt_positive)
    if average == 'micro':
        tp_sum = class_correct.sum()
        pred_sum = pred_positive.sum()
        gt_sum = gt_positive.sum()
    else:
        tp_sum = class_correct.sum(0)
        pred_sum = pred_positive.sum(0)
        gt_sum = gt_positive.sum(0)

    precision = tp_sum / torch.clamp(pred_sum, min=1).float() * 100
    recall = tp_sum / torch.clamp(gt_sum, min=1).float() * 100
    f1_score = 2 * precision * recall / torch.clamp(
        precision + recall, min=torch.finfo(torch.float32).eps)
    if average in ['macro', 'micro']:
        precision = precision.mean(0)
        recall = recall.mean(0)
        f1_score = f1_score.mean(0)
        support = gt_sum.sum(0)
    else:
        support = gt_sum
    return precision, recall, f1_score, support


@METRICS.register_module()
class Loss(BaseMetric):
    r"""Loss evaluation metric.

    For either binary classification or multi-class classification, the
    loss is derived from model:

    .. math::

        \text{Loss} = \frac{1}{N}\sum_i^N(l_i)

    Args:
        task (str): Task name used for collecting loss
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """
    default_prefix: Optional[str] = 'loss'

    def __init__(self,
                 task: str = 'classification',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.task = task

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
            if f'{self.task}_loss' in data_sample:
                result[self.task] = data_sample[self.task + '_loss'].cpu()
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

        # concat
        loss = torch.cat([res[self.task] for res in results])
        loss = self.calculate(loss)
        metrics[self.task] = loss.item()

        return metrics

    @staticmethod
    def calculate(
            loss: Union[torch.Tensor, np.ndarray, Sequence],
    ) -> Union[torch.Tensor, List[List[torch.Tensor]]]:
        """Calculate the loss.

        Args:
            loss (torch.Tensor | np.ndarray | Sequence): The loss
                results. It can be scalars (N, )

        Returns:
            torch.Tensor | List[List[torch.Tensor]]: Loss.
        """

        loss = to_tensor(loss)
        num = loss.size(0)
        loss = loss.sum()
        loss = loss.mul_(1. / num)
        return loss


@METRICS.register_module()
class FeaturesDistribution(BaseMetric):
    default_prefix: Optional[str] = 'feature'

    def __init__(self,
                 classes: Sequence[str],
                 vis_dim: int = 2,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.classes = classes
        self.num_classes = len(classes)
        if vis_dim > 3:
            raise Warning(
                'Currently, only support visualization of features with dimension <=3. So if vis_dim is >3, it'
                'will be truncated as 3.')
            vis_dim = 3

        self.vis_dim = vis_dim

    def process(self, data_batch, data_samples: Sequence[dict]):
        for data_sample in data_samples:
            result = dict()
            result['feature'] = data_sample['feature'].cpu()
            result['gt_label'] = data_sample['gt_label'].cpu()
            num_classes = self.num_classes or data_sample.get('num_classes')
            result['num_classes'] = num_classes
            result['vid_dim'] = self.vis_dim
            self.results.append(result)

    def compute_metrics(self, results: list):
        metrics = {}
        target = torch.cat([res['gt_label']] for res in results)
        feature = torch.cat([res['feature']] for res in results)
        feature, center, silhouette_avg, sample_silhouette_values = self.calculate(feature, target,
                                                                                   results[0]['num_classes'],
                                                                                   results[0]['vis_dim'])
        metrics['silhouette'] = silhouette_avg
        return metrics

    @staticmethod
    def calculate(
            feature: Union[torch.Tensor, np.ndarray, Sequence],
            target: Union[torch.Tensor, np.ndarray, Sequence],
            num_classes: int,
            vis_dim: int = 2,
    ):

        feature, center, silhouette_avg, sample_silhouette_values = _feature_distribution(feature, target, num_classes,
                                                                                          vis_dim)
        return feature, center, silhouette_avg, sample_silhouette_values


@METRICS.register_module()
class ROC(BaseMetric):
    default_prefix: Optional[str] = 'ROC'

    def __init__(self,
                 classes: Sequence[str],
                 mode: Optional[str] = 'ovr',
                 average: Optional[str] = 'macro',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.classes = classes
        self.num_classes = len(classes)
        self.mode = mode
        self.average = average
        if self.mode == 'ovo':
            self.average = 'macro'

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
            if 'pred_score' in data_sample:
                result['pred_score'] = data_sample['pred_score'].cpu()
            else:
                num_classes = self.num_classes or data_sample.get(
                    'num_classes')
                assert num_classes is not None, \
                    'The `num_classes` must be specified if no `pred_score`.'
                result['pred_label'] = data_sample['pred_label'].cpu()
                result['num_classes'] = num_classes
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

        # concat
        target = torch.cat([res['gt_label'] for res in results])
        if 'pred_score' in results[0]:
            pred = torch.stack([res['pred_score'] for res in results])
        else:
            pred = torch.cat([res['pred_label'] for res in results])
        roc = self.calculate(pred, target, results[0]['num_classes'])

        metrics['auc'] = roc[self.mode]['auc'][self.average]

    @staticmethod
    def calculate(
            pred: Union[torch.Tensor, np.ndarray, Sequence],
            target: Union[torch.Tensor, np.ndarray, Sequence],
            num_classes: Optional[int] = None,
    ):

        gt_positive = F.one_hot(target.flatten(), num_classes)
        if pred.ndim == 1:
            assert num_classes is not None, \
                'Please specify the `num_classes` if the `pred` is labels ' \
                'intead of scores.'
            pred_positive = F.one_hot(pred.to(torch.int64), num_classes)
        else:
            pred_positive = pred

        roc = _roc_ovr_ovo(pred_positive, gt_positive, target, num_classes)

        return roc


@METRICS.register_module()
class Accuracy(BaseMetric):
    r"""Accuracy evaluation metric.

    For either binary classification or multi-class classification, the
    accuracy is the fraction of correct predictions in all predictions:

    .. math::

        \text{Accuracy} = \frac{N_{\text{correct}}}{N_{\text{all}}}

    Args:
        topk (int | Sequence[int]): If the ground truth label matches one of
            the best **k** predictions, the sample will be regard as a positive
            prediction. If the parameter is a tuple, all of top-k accuracy will
            be calculated and outputted together. Defaults to 1.
        thrs (Sequence[float | None] | float | None): If a float, predictions
            with score lower than the threshold will be regard as the negative
            prediction. If None, not apply threshold. If the parameter is a
            tuple, accuracy based on all thresholds will be calculated and
            outputted together. Defaults to 0.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.

    Examples:
        >>> import torch
        >>> from csrr.evaluation import Accuracy
        >>> # -------------------- The Basic Usage --------------------
        >>> y_pred = [0, 2, 1, 3]
        >>> y_true = [0, 1, 2, 3]
        >>> Accuracy.calculate(y_pred, y_true)
        tensor([50.])
        >>> # Calculate the top1 and top5 accuracy.
        >>> y_score = torch.rand((1000, 10))
        >>> y_true = torch.zeros((1000, ))
        >>> Accuracy.calculate(y_score, y_true, topk=(1, 5))
        [[tensor([9.9000])], [tensor([51.5000])]]
        >>>
        >>> # ------------------- Use with Evalutor -------------------
        >>> from csrr.structures import DataSample
        >>> from mmengine.evaluator import Evaluator
        >>> data_samples = [
        ...     DataSample().set_gt_label(0).set_pred_score(torch.rand(10))
        ...     for i in range(1000)
        ... ]
        >>> evaluator = Evaluator(metrics=Accuracy(topk=(1, 5)))
        >>> evaluator.process(data_samples)
        >>> evaluator.evaluate(1000)
        {
            'accuracy/top1': 9.300000190734863,
            'accuracy/top5': 51.20000076293945
        }
    """
    default_prefix: Optional[str] = 'accuracy'

    def __init__(self,
                 topk: Union[int, Sequence[int]] = (1,),
                 thrs: Union[float, Sequence[Union[float, None]], None] = 0.,
                 collect_device: str = 'cpu',
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
            if 'pred_score' in data_sample:
                result['pred_score'] = data_sample['pred_score'].cpu()
            else:
                result['pred_label'] = data_sample['pred_label'].cpu()
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

        # concat
        target = torch.cat([res['gt_label'] for res in results])
        if 'pred_score' in results[0]:
            pred = torch.stack([res['pred_score'] for res in results])

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
                    name = f'top{k}'
                    if multi_thrs:
                        name += '_no-thr' if thr is None else f'_thr-{thr:.2f}'
                    metrics[name] = acc[i][j].item()
        else:
            # If only label in the `pred_label`.
            pred = torch.cat([res['pred_label'] for res in results])
            acc = self.calculate(pred, target, self.topk, self.thrs)
            metrics['top1'] = acc.item()

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


@METRICS.register_module()
class SingleLabelMetric(BaseMetric):
    r"""A collection of precision, recall, f1-score and support for
    single-label tasks.

    The collection of metrics is for single-label multi-class classification.
    And all these metrics are based on the confusion matrix of every category:

    .. image:: ../../_static/image/confusion-matrix.png
       :width: 60%
       :align: center

    All metrics can be formulated use variables above:

    **Precision** is the fraction of correct predictions in all predictions:

    .. math::
        \text{Precision} = \frac{TP}{TP+FP}

    **Recall** is the fraction of correct predictions in all targets:

    .. math::
        \text{Recall} = \frac{TP}{TP+FN}

    **F1-score** is the harmonic mean of the precision and recall:

    .. math::
        \text{F1-score} = \frac{2\times\text{Recall}\times\text{Precision}}{\text{Recall}+\text{Precision}}

    **Support** is the number of samples:

    .. math::
        \text{Support} = TP + TN + FN + FP

    Args:
        thrs (Sequence[float | None] | float | None): If a float, predictions
            with score lower than the threshold will be regard as the negative
            prediction. If None, only the top-1 prediction will be regard as
            the positive prediction. If the parameter is a tuple, accuracy
            based on all thresholds will be calculated and outputted together.
            Defaults to 0.
        items (Sequence[str]): The detailed metric items to evaluate, select
            from "precision", "recall", "f1-score" and "support".
            Defaults to ``('precision', 'recall', 'f1-score')``.
        average (str | None): How to calculate the final metrics from the
            confusion matrix of every category. It supports three modes:

            - `"macro"`: Calculate metrics for each category, and calculate
              the mean value over all categories.
            - `"micro"`: Average the confusion matrix over all categories and
              calculate metrics on the mean confusion matrix.
            - `None`: Calculate metrics of every category and output directly.

            Defaults to "macro".
        num_classes (int, optional): The number of classes. Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.

    Examples:
        >>> import torch
        >>> from csrr.evaluation import SingleLabelMetric
        >>> # -------------------- The Basic Usage --------------------
        >>> y_pred = [0, 1, 1, 3]
        >>> y_true = [0, 2, 1, 3]
        >>> # Output precision, recall, f1-score and support.
        >>> SingleLabelMetric.calculate(y_pred, y_true, num_classes=4)
        (tensor(62.5000), tensor(75.), tensor(66.6667), tensor(4))
        >>> # Calculate with different thresholds.
        >>> y_score = torch.rand((1000, 10))
        >>> y_true = torch.zeros((1000, ))
        >>> SingleLabelMetric.calculate(y_score, y_true, thrs=(0., 0.9))
        [(tensor(10.), tensor(0.9500), tensor(1.7352), tensor(1000)),
         (tensor(10.), tensor(0.5500), tensor(1.0427), tensor(1000))]
        >>>
        >>> # ------------------- Use with Evalutor -------------------
        >>> from csrr.structures import DataSample
        >>> from mmengine.evaluator import Evaluator
        >>> data_samples = [
        ...     DataSample().set_gt_label(i%5).set_pred_score(torch.rand(5))
        ...     for i in range(1000)
        ... ]
        >>> evaluator = Evaluator(metrics=SingleLabelMetric())
        >>> evaluator.process(data_samples)
        >>> evaluator.evaluate(1000)
        {'single-label/precision': 19.650691986083984,
         'single-label/recall': 19.600000381469727,
         'single-label/f1-score': 19.619548797607422}
        >>> # Evaluate on each class
        >>> evaluator = Evaluator(metrics=SingleLabelMetric(average=None))
        >>> evaluator.process(data_samples)
        >>> evaluator.evaluate(1000)
        {
            'single-label/precision_classwise': [21.1, 18.7, 17.8, 19.4, 16.1],
            'single-label/recall_classwise': [18.5, 18.5, 17.0, 20.0, 18.0],
            'single-label/f1-score_classwise': [19.7, 18.6, 17.1, 19.7, 17.0]
        }
    """  # noqa: E501
    default_prefix: Optional[str] = 'single-label'

    def __init__(self,
                 thrs: Union[float, Sequence[Union[float, None]], None] = 0.,
                 items: Sequence[str] = ('precision', 'recall', 'f1-score'),
                 average: Optional[str] = 'macro',
                 num_classes: Optional[int] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        if isinstance(thrs, float) or thrs is None:
            self.thrs = (thrs,)
        else:
            self.thrs = tuple(thrs)

        for item in items:
            assert item in ['precision', 'recall', 'f1-score', 'support'], \
                f'The metric {item} is not supported by `SingleLabelMetric`,' \
                ' please specify from "precision", "recall", "f1-score" and ' \
                '"support".'
        self.items = tuple(items)
        self.average = average
        self.num_classes = num_classes

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
            if 'pred_score' in data_sample:
                result['pred_score'] = data_sample['pred_score'].cpu()
            else:
                num_classes = self.num_classes or data_sample.get(
                    'num_classes')
                assert num_classes is not None, \
                    'The `num_classes` must be specified if no `pred_score`.'
                result['pred_label'] = data_sample['pred_label'].cpu()
                result['num_classes'] = num_classes
            result['gt_label'] = data_sample['gt_label'].cpu()
            # Save the result to `self.results`.
            self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # NOTICE: don't access `self.results` from the method. `self.results`
        # are a list of results from multiple batch, while the input `results`
        # are the collected results.
        metrics = {}

        def pack_results(precision, recall, f1_score, support):
            single_metrics = {}
            if 'precision' in self.items:
                single_metrics['precision'] = precision
            if 'recall' in self.items:
                single_metrics['recall'] = recall
            if 'f1-score' in self.items:
                single_metrics['f1-score'] = f1_score
            if 'support' in self.items:
                single_metrics['support'] = support
            return single_metrics

        # concat
        target = torch.cat([res['gt_label'] for res in results])
        if 'pred_score' in results[0]:
            pred = torch.stack([res['pred_score'] for res in results])
            metrics_list = self.calculate(
                pred, target, thrs=self.thrs, average=self.average)

            multi_thrs = len(self.thrs) > 1
            for i, thr in enumerate(self.thrs):
                if multi_thrs:
                    suffix = '_no-thr' if thr is None else f'_thr-{thr:.2f}'
                else:
                    suffix = ''

                for k, v in pack_results(*metrics_list[i]).items():
                    metrics[k + suffix] = v
        else:
            # If only label in the `pred_label`.
            pred = torch.cat([res['pred_label'] for res in results])
            res = self.calculate(
                pred,
                target,
                average=self.average,
                num_classes=results[0]['num_classes'])
            metrics = pack_results(*res)

        result_metrics = dict()
        for k, v in metrics.items():

            if self.average is None:
                result_metrics[k + '_classwise'] = v.cpu().detach().tolist()
            elif self.average == 'micro':
                result_metrics[k + f'_{self.average}'] = v.item()
            else:
                result_metrics[k] = v.item()

        return result_metrics

    @staticmethod
    def calculate(
            pred: Union[torch.Tensor, np.ndarray, Sequence],
            target: Union[torch.Tensor, np.ndarray, Sequence],
            thrs: Sequence[Union[float, None]] = (0.,),
            average: Optional[str] = 'macro',
            num_classes: Optional[int] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Calculate the precision, recall, f1-score and support.

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
            average (str | None): How to calculate the final metrics from
                the confusion matrix of every category. It supports three
                modes:

                - `"macro"`: Calculate metrics for each category, and calculate
                  the mean value over all categories.
                - `"micro"`: Average the confusion matrix over all categories
                  and calculate metrics on the mean confusion matrix.
                - `None`: Calculate metrics of every category and output
                  directly.

                Defaults to "macro".
            num_classes (Optional, int): The number of classes. If the ``pred``
                is label instead of scores, this argument is required.
                Defaults to None.

        Returns:
            Tuple: The tuple contains precision, recall and f1-score.
            And the type of each item is:

            - torch.Tensor: If the ``pred`` is a sequence of label instead of
              score (number of dimensions is 1). Only returns a tensor for
              each metric. The shape is (1, ) if ``classwise`` is False, and
              (C, ) if ``classwise`` is True.
            - List[torch.Tensor]: If the ``pred`` is a sequence of score
              (number of dimensions is 2). Return the metrics on each ``thrs``.
              The shape of tensor is (1, ) if ``classwise`` is False, and (C, )
              if ``classwise`` is True.
        """
        average_options = ['micro', 'macro', None]
        assert average in average_options, 'Invalid `average` argument, ' \
                                           f'please specify from {average_options}.'

        pred = to_tensor(pred)
        target = to_tensor(target).to(torch.int64)
        assert pred.size(0) == target.size(0), \
            f"The size of pred ({pred.size(0)}) doesn't match " \
            f'the target ({target.size(0)}).'

        if pred.ndim == 1:
            assert num_classes is not None, \
                'Please specify the `num_classes` if the `pred` is labels ' \
                'intead of scores.'
            gt_positive = F.one_hot(target.flatten(), num_classes)
            pred_positive = F.one_hot(pred.to(torch.int64), num_classes)
            return _precision_recall_f1_support(pred_positive, gt_positive,
                                                average)
        else:
            # For pred score, calculate on all thresholds.
            num_classes = pred.size(1)
            pred_score, pred_label = torch.topk(pred, k=1)
            pred_score = pred_score.flatten()
            pred_label = pred_label.flatten()

            gt_positive = F.one_hot(target.flatten(), num_classes)

            results = []
            for thr in thrs:
                pred_positive = F.one_hot(pred_label, num_classes)
                if thr is not None:
                    pred_positive[pred_score <= thr] = 0
                results.append(
                    _precision_recall_f1_support(pred_positive, gt_positive,
                                                 average))

            return results


@METRICS.register_module()
class ConfusionMatrix(BaseMetric):
    r"""A metric to calculate confusion matrix for single-label tasks.

    Args:
        num_classes (int, optional): The number of classes. Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.

    Examples:

        1. The basic usage.

        >>> import torch
        >>> from csrr.evaluation import ConfusionMatrix
        >>> y_pred = [0, 1, 1, 3]
        >>> y_true = [0, 2, 1, 3]
        >>> ConfusionMatrix.calculate(y_pred, y_true, num_classes=4)
        tensor([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]])
        >>> # plot the confusion matrix
        >>> import matplotlib.pyplot as plt
        >>> y_score = torch.rand((1000, 10))
        >>> y_true = torch.randint(10, (1000, ))
        >>> matrix = ConfusionMatrix.calculate(y_score, y_true)
        >>> ConfusionMatrix().plot(matrix)
        >>> plt.show()

        2. In the config file

        .. code:: python

            val_evaluator = dict(type='ConfusionMatrix')
            test_evaluator = dict(type='ConfusionMatrix')
    """  # noqa: E501
    default_prefix = 'confusion_matrix'

    def __init__(self,
                 num_classes: Optional[int] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device, prefix)

        self.num_classes = num_classes

    def process(self, data_batch, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            if 'pred_score' in data_sample:
                pred_score = data_sample['pred_score']
                pred_label = pred_score.argmax(dim=0, keepdim=True)
                self.num_classes = pred_score.size(0)
            else:
                pred_label = data_sample['pred_label']

            self.results.append({
                'pred_label': pred_label,
                'gt_label': data_sample['gt_label'],
            })

    def compute_metrics(self, results: list) -> dict:
        pred_labels = []
        gt_labels = []
        for result in results:
            pred_labels.append(result['pred_label'])
            gt_labels.append(result['gt_label'])
        confusion_matrix = ConfusionMatrix.calculate(
            torch.cat(pred_labels),
            torch.cat(gt_labels),
            num_classes=self.num_classes)
        return {'result': confusion_matrix}

    @staticmethod
    def calculate(pred, target, num_classes=None) -> dict:
        """Calculate the confusion matrix for single-label task.

        Args:
            pred (torch.Tensor | np.ndarray | Sequence): The prediction
                results. It can be labels (N, ), or scores of every
                class (N, C).
            target (torch.Tensor | np.ndarray | Sequence): The target of
                each prediction with shape (N, ).
            num_classes (Optional, int): The number of classes. If the ``pred``
                is label instead of scores, this argument is required.
                Defaults to None.

        Returns:
            torch.Tensor: The confusion matrix.
        """
        pred = to_tensor(pred)
        target_label = to_tensor(target).int()

        assert pred.size(0) == target_label.size(0), \
            f"The size of pred ({pred.size(0)}) doesn't match " \
            f'the target ({target_label.size(0)}).'
        assert target_label.ndim == 1

        if pred.ndim == 1:
            assert num_classes is not None, \
                'Please specify the `num_classes` if the `pred` is labels ' \
                'intead of scores.'
            pred_label = pred
        else:
            num_classes = num_classes or pred.size(1)
            pred_label = torch.argmax(pred, dim=1).flatten()

        with torch.no_grad():
            indices = num_classes * target_label + pred_label
            matrix = torch.bincount(indices, minlength=num_classes ** 2)
            matrix = matrix.reshape(num_classes, num_classes)

        return matrix

    @staticmethod
    def plot(confusion_matrix: torch.Tensor,
             include_values: bool = False,
             cmap: str = 'viridis',
             classes: Optional[List[str]] = None,
             colorbar: bool = True,
             show: bool = True):
        """Draw a confusion matrix by matplotlib.

        Modified from `Scikit-Learn
        <https://github.com/scikit-learn/scikit-learn/blob/dc580a8ef/sklearn/metrics/_plot/confusion_matrix.py#L81>`_

        Args:
            confusion_matrix (torch.Tensor): The confusion matrix to draw.
            include_values (bool): Whether to draw the values in the figure.
                Defaults to False.
            cmap (str): The color map to use. Defaults to use "viridis".
            classes (list[str], optional): The names of categories.
                Defaults to None, which means to use index number.
            colorbar (bool): Whether to show the colorbar. Defaults to True.
            show (bool): Whether to show the figure immediately.
                Defaults to True.
        """  # noqa: E501
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 10))

        num_classes = confusion_matrix.size(0)

        im_ = ax.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
        text_ = None
        cmap_min, cmap_max = im_.cmap(0), im_.cmap(1.0)

        if include_values:
            text_ = np.empty_like(confusion_matrix, dtype=object)

            # print text with appropriate color depending on background
            thresh = (confusion_matrix.max() + confusion_matrix.min()) / 2.0

            for i, j in product(range(num_classes), range(num_classes)):
                color = cmap_max if confusion_matrix[i,
                j] < thresh else cmap_min

                text_cm = format(confusion_matrix[i, j], '.2g')
                text_d = format(confusion_matrix[i, j], 'd')
                if len(text_d) < len(text_cm):
                    text_cm = text_d

                text_[i, j] = ax.text(
                    j, i, text_cm, ha='center', va='center', color=color)

        display_labels = classes or np.arange(num_classes)

        if colorbar:
            fig.colorbar(im_, ax=ax)
        ax.set(
            xticks=np.arange(num_classes),
            yticks=np.arange(num_classes),
            xticklabels=display_labels,
            yticklabels=display_labels,
            ylabel='True label',
            xlabel='Predicted label',
        )
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        ax.set_ylim((num_classes - 0.5, -0.5))
        # Automatically rotate the x labels.
        fig.autofmt_xdate(ha='center')

        if show:
            plt.show()
        return fig
