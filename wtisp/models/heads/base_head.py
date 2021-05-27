from abc import ABCMeta, abstractmethod

import torch.nn as nn


class BaseHead(nn.Module, metaclass=ABCMeta):
    """Base class for ClassifierHeads."""

    def __init__(self):
        super(BaseHead, self).__init__()

    @abstractmethod
    def loss(self, x, **kwargs):
        """Compute losses of the head."""
        pass

    def forward_train(self,
                      x,
                      **kwargs):
        """
        Args:
            x (Tensor or dict[str, Tensor]): Features from Backbone.
            labels (Tensor): classification labels

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
        """
        loss_inputs = self(x)
        losses = self.loss(loss_inputs, **kwargs)

        return losses
