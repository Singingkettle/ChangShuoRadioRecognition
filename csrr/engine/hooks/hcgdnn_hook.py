from typing import Dict, Optional

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from csrr.models.heads import HCGDNNHead
from csrr.registry import HOOKS


@HOOKS.register_module()
class HCGDNNHook(Hook):
    """A Hook to update fusion weights in HCGDNNHead, which are derived from the validation set.
    """

    priority = 'HIGHEST'

    def __init__(self, outputs=['cnn', 'gru1', 'gru2']) -> None:
        self.outputs = outputs

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        """We recover source model's parameter from ema model after validation.

        Args:
            runner (Runner): The runner of the validation process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on validation dataset. The keys are the names of the
                metrics, and the values are corresponding results.
        """

        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        if (hasattr(model, 'head')
                and not isinstance(model.head, HCGDNNHead)):
            raise ValueError(
                'Hook ``HCGDNNHook`` could only be used '
                f'for ``HCGDNNHead``, but get {type(model.head)}')

        weights = dict()

        for head_name in self.outputs:
            weights[head_name] = metrics[f'weights/{head_name}']

        runner.logger.info('Update HCGDNN Fusion Weights using Validation Set.')
        model.head.set_weights(weights)
