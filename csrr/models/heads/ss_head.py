import torch

from .classification_head import ClassificationHead
from ..builder import HEADS


@HEADS.register_module()
class SSHead(ClassificationHead):
    def __init__(self, num_classes, in_features=10560, out_features=256,
                 loss_cls=None):
        super(SSHead, self).__init__(num_classes, in_features, out_features, loss_cls)

    def loss(self, inputs, labels, weight=None, **kwargs):
        labels = labels.view(-1)
        loss_Final = self.loss_cls(inputs, labels, weight=weight)
        return dict(loss_Final=loss_Final)

    def forward(self, inputs, vis_fea=False, is_test=False):
        pre = self.classifier(inputs)
        if is_test:
            pre = torch.softmax(pre, dim=1)
        return pre
