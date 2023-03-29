from .classification_head import ClassificationHead
from ..builder import HEADS


@HEADS.register_module()
class DSCLDNNHead(ClassificationHead):
    def __init__(self, num_classes, in_size=10560, loss_cls=None, init_cfg=None):
        super(DSCLDNNHead, self).__init__(num_classes, in_size=in_size, loss_cls=loss_cls, is_shallow=True,
                                          init_cfg=init_cfg)

    def loss(self, x, targets, weight=None, **kwargs):
        loss_cls = self.loss_cls(x, targets['modulations'], weight=weight)
        return dict(loss_cls=loss_cls)
