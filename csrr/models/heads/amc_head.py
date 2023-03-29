from .classification_head import ClassificationHead
from ..builder import HEADS


@HEADS.register_module()
class AMCHead(ClassificationHead):
    def __init__(self, num_classes, in_size=10560, out_size=256, loss_cls=None, init_cfg=None):
        super(AMCHead, self).__init__(num_classes, in_size=in_size, out_size=out_size, loss_cls=loss_cls,
                                      init_cfg=init_cfg)

    def loss(self, x, targets, weight=None, **kwargs):
        loss_cls = self.loss_cls(x, targets['modulations'], weight=weight)
        return dict(loss_cls=loss_cls)
