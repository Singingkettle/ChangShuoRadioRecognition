import numpy as np
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from .builder import DATASETS
from ..common import print_log


@DATASETS.register_module()
class ConcatAMCDataset(_ConcatDataset):
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
        super(ConcatAMCDataset, self).__init__(datasets)
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
