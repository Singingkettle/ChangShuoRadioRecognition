import torch.nn as nn

from .classification_head import ClassificationHead
from ..builder import HEADS, build_loss


@HEADS.register_module()
class AMCHead(ClassificationHead):
    def __init__(self, num_classes, in_features=10560, out_features=256, loss_cls=None):
        super(AMCHead, self).__init__(num_classes, in_features, out_features, loss_cls)

    def forward_train(self,
                      inputs,
                      modulations,
                      **kwargs):
        """
        Args:
            inputs (Tensor or dict[str, Tensor]): Features from Backbone.
            modulations (Tensor): modulation labels

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
        """
        loss_inputs = self(inputs)
        losses = self.loss(loss_inputs, modulations, **kwargs)

        return losses
