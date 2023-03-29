from .base_classifier import BaseClassifier

from ..builder import METHODS


@METHODS.register_module()
class AlexNet(BaseClassifier):
    def __init__(self, backbone, classifier_head, vis_fea=False, init_cfg=None):
        super(AlexNet, self).__init__(backbone, classifier_head, vis_fea, 'AlexNet', init_cfg)
