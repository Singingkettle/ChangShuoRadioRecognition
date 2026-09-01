"""Constant-memory validation metrics for large AMC datasets."""

from typing import Optional, Sequence, Union

import torch
from mmengine.dist import all_reduce, get_world_size
from mmengine.evaluator import BaseMetric

from csrr.registry import METRICS


class _StreamingMetric(BaseMetric):
    def __init__(self, expected_samples: int, expected_world_size: int,
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device='cpu', prefix=prefix)
        if expected_samples <= 0 or expected_world_size <= 0:
            raise ValueError('expected sample and world sizes must be positive')
        if expected_samples % expected_world_size:
            raise ValueError('validation split would require sampler padding')
        self.expected_samples = expected_samples
        self.expected_world_size = expected_world_size

    def _validate(self, size: int) -> None:
        if size != self.expected_samples:
            raise RuntimeError(
                f'validation size differs: {size} != {self.expected_samples}')
        if get_world_size() != self.expected_world_size:
            raise RuntimeError(
                'world size differs from the declared validation protocol')
        if not self.results:
            raise RuntimeError('streaming metric received no batches')

    def _with_prefix(self, values):
        if self.prefix:
            return {f'{self.prefix}/{key}': value
                    for key, value in values.items()}
        return values

    def compute_metrics(self, results):
        raise NotImplementedError('streaming metrics reduce in evaluate()')


@METRICS.register_module()
class StreamingAccuracy(_StreamingMetric):
    """Exact top-1 accuracy reduced from correct and sample counts."""

    default_prefix = 'accuracy'

    def __init__(self, expected_samples: int, expected_world_size: int,
                 topk: Union[int, Sequence[int]] = (1,),
                 thrs: Union[float, Sequence[float], None] = 0.0,
                 prefix: Optional[str] = None) -> None:
        super().__init__(expected_samples, expected_world_size, prefix)
        topk = (topk,) if isinstance(topk, int) else tuple(topk)
        thrs = (thrs,) if isinstance(thrs, (float, int)) or thrs is None \
            else tuple(thrs)
        if topk != (1,) or thrs != (0.0,):
            raise ValueError(
                'StreamingAccuracy supports top-1 with threshold 0.0 only')

    def process(self, data_batch, data_samples) -> None:
        if not data_samples:
            raise RuntimeError('accuracy batch is empty')
        scores = torch.stack(
            [sample['pred_score'].detach() for sample in data_samples])
        labels = torch.cat(
            [sample['gt_label'].detach() for sample in data_samples])
        labels = labels.to(device=scores.device, dtype=torch.int64)
        if scores.ndim != 2 or scores.shape[0] != labels.numel():
            raise RuntimeError('accuracy score/label shape differs')
        correct = scores.argmax(dim=1).eq(labels).sum(dtype=torch.int64)
        count = torch.tensor(labels.numel(), device=scores.device,
                             dtype=torch.int64)
        self.results.append((correct, count))

    def evaluate(self, size: int):
        self._validate(size)
        correct = torch.stack([item[0] for item in self.results]).sum()
        count = torch.stack([item[1] for item in self.results]).sum()
        all_reduce(correct)
        all_reduce(count)
        if int(count.item()) != self.expected_samples:
            raise RuntimeError('accuracy sample count differs')
        value = correct.float().mul(100.0 / self.expected_samples).item()
        self.results.clear()
        return self._with_prefix({'top1': value})


@METRICS.register_module()
class StreamingLoss(_StreamingMetric):
    """Mean per-sample diagnostic loss reduced from sum and count."""

    default_prefix = 'loss'

    def __init__(self, expected_samples: int, expected_world_size: int,
                 task: str = 'classification',
                 prefix: Optional[str] = None) -> None:
        super().__init__(expected_samples, expected_world_size, prefix)
        if not task:
            raise ValueError('task must be non-empty')
        self.task = task

    def process(self, data_batch, data_samples) -> None:
        if not data_samples:
            raise RuntimeError('loss batch is empty')
        key = f'{self.task}_loss'
        if any(key not in sample for sample in data_samples):
            raise RuntimeError(f'loss batch lacks {key}')
        losses = torch.cat(
            [sample[key].detach().reshape(-1) for sample in data_samples])
        if losses.numel() != len(data_samples):
            raise RuntimeError('loss is not one scalar per sample')
        count = torch.tensor(losses.numel(), device=losses.device,
                             dtype=torch.int64)
        self.results.append((losses.sum(dtype=torch.float64), count))

    def evaluate(self, size: int):
        self._validate(size)
        loss_sum = torch.stack([item[0] for item in self.results]).sum()
        count = torch.stack([item[1] for item in self.results]).sum()
        all_reduce(loss_sum)
        all_reduce(count)
        if int(count.item()) != self.expected_samples:
            raise RuntimeError('loss sample count differs')
        value = loss_sum.div(self.expected_samples).item()
        self.results.clear()
        return self._with_prefix({self.task: value})
