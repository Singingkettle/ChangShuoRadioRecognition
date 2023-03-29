import torch

from .base_head import BaseHead
from ..builder import HEADS, build_loss
from ...ops import Euclidean


@HEADS.register_module()
class CenterHead(BaseHead):
    def __init__(self, num_classes, in_size=10560, loss_center=None, init_cfg=None):
        super(CenterHead, self).__init__(init_cfg)
        if loss_center is None:
            loss_center = dict(
                type='CenterLoss',
            )
        self.num_classes = num_classes
        self.loss_center = build_loss(loss_center)
        self.center = Euclidean(in_size, num_classes)

    def loss(self, inputs, targets, weight=None, **kwargs):
        loss_center = self.loss_center(inputs, targets, weight=weight)
        return dict(loss_center=loss_center)

    def forward(self, x, vis_fea=False, is_test=False):
        x = self.center(x)
        x = torch.pow(x, 2)
        return x
