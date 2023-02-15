from abc import ABCMeta, abstractmethod

import torch.nn as nn


class BaseHead(nn.Module, metaclass=ABCMeta):
    """Base class for ClassifierHeads."""

    def __init__(self):
        super(BaseHead, self).__init__()

    @abstractmethod
    def loss(self, inputs, labels, **kwargs):
        """Compute losses of the head."""
        pass

    def forward_train(self,
                      inputs,
                      labels,
                      **kwargs):
        """
        Args:
            inputs (Tensor or dict[str, Tensor]): Features from Backbone.
            labels (Tensor): classification labels

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
        """
        loss_inputs = self(inputs)
        losses = self.loss(loss_inputs, labels, **kwargs)

        return losses
