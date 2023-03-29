from abc import ABCMeta, abstractmethod

from ...runner import BaseModule


class BaseHead(BaseModule, metaclass=ABCMeta):
    """Base class for ClassifierHeads."""

    def __init__(self, init_cfg=None):
        if init_cfg is None:
            init_cfg = [
                dict(type='Kaiming', layer='Linear', mode='fan_in', nonlinearity='relu'),
            ]
        super(BaseHead, self).__init__(init_cfg)

    @abstractmethod
    def loss(self, inputs, targets, **kwargs):
        """Compute losses of the head."""
        pass

    def forward_train(self,
                      inputs,
                      targets,
                      **kwargs):
        """
        Args:
            inputs (Tensor or dict[str, Tensor]): Features from Backbone.
            targets (Tensor or dict[str, Tensor]): Training Targets for Supervision

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
        """
        loss_inputs = self(inputs)
        losses = self.loss(loss_inputs, targets, **kwargs)

        return losses
