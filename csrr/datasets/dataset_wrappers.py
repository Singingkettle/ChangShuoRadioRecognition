import copy
import copy
import warnings
from typing import List, Union

import numpy as np
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from .builder import DATASETS, build_dataset
from ..common import print_log


@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    concat the group flag for image aspect ratio.

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
        separate_eval (bool): Whether to evaluate the results
            separately if it is used as validation dataset.
            Defaults to True.
    """

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__(datasets)
        self.CLASSES = datasets[0].CLASSES
        if hasattr(datasets[0], 'flag'):
            flags = []
            for i in range(0, len(datasets)):
                flags.append(datasets[i].flag)
            self.flag = np.concatenate(flags)

    def evaluate(self, results, logger=None, **kwargs):
        """Evaluate the results.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str: float]: AP results of the total dataset or each separate
            dataset if `self.separate_eval=True`.
        """
        assert len(results) == self.cumulative_sizes[-1], \
            ('Dataset and results have different sizes: '
             f'{self.cumulative_sizes[-1]} v.s. {len(results)}')

        # Check whether all the datasets support evaluation
        for dataset in self.datasets:
            assert hasattr(dataset, 'evaluate'), \
                f'{type(dataset)} does not implement evaluate function'

        dataset_idx = -1
        total_eval_results = dict()
        for size, dataset in zip(self.cumulative_sizes, self.datasets):
            start_idx = 0 if dataset_idx == -1 else \
                self.cumulative_sizes[dataset_idx]
            end_idx = self.cumulative_sizes[dataset_idx + 1]

            results_per_dataset = results[start_idx:end_idx]
            print_log(
                f'\nEvaluateing {dataset.ann_file} with '
                f'{len(results_per_dataset)} items now',
                logger=logger)

            eval_results_per_dataset = dataset.evaluate(
                results_per_dataset, logger=logger, **kwargs)
            dataset_idx += 1
            for k, v in eval_results_per_dataset.items():
                total_eval_results.update({f'{dataset_idx}_{k}': v})

        return total_eval_results


@DATASETS.register_module()
class RepeatDataset:
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Note:
        ``RepeatDataset`` should not inherit from ``BaseDataset`` since
        ``get_subset`` and ``get_subset_`` could produce ambiguous meaning
        sub-dataset which conflicts with original dataset. If you want to use
        a sub-dataset of ``RepeatDataset``, you should set ``indices``
        arguments for wrapped dataset which inherit from ``BaseDataset``.

    Args:
        dataset (BaseDataset or dict): The dataset to be repeated.
        times (int): Repeat times.
        lazy_init (bool): Whether to load annotation during
            instantiation. Defaults to False.
    """

    def __init__(self,
                 dataset,
                 times: int):

        self.times = times
        self.dataset = build_dataset(dataset)
        self.CLASSES = self.dataset.CLASSES
        if hasattr(self.dataset, 'flag'):
            flags = []
            for i in range(self.times):
                flags.append(self.dataset.flag)
            self.flag = np.concatenate(flags)

    def __len__(self):
        return self.times * len(self.dataset)

    def __getitem__(self, idx):
        idx = idx % len(self.dataset)
        return self.dataset[idx]