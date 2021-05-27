from .base_head import BaseHead
from ..builder import HEADS, build_loss


@HEADS.register_module()
class SeparatorHead(BaseHead):
    def __init__(self, loss_reg):
        super(SeparatorHead, self).__init__()
        self.loss_reg = build_loss(loss_reg)

    def init_weights(self):
        pass

    def loss(self, x, gt_signals=None):
        loss_reg = self.loss_reg(x, gt_signals)

        return dict(loss_reg=loss_reg)

    def forward(self, x):
        return x
