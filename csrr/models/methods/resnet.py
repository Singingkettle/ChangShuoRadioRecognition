from .base_classifier import BaseClassifier

from ..builder import METHODS


@METHODS.register_module()
class ResNet(BaseClassifier):
    def __init__(self, backbone, classifier_head, vis_fea=False):
        super(ResNet, self).__init__(backbone, classifier_head, vis_fea, 'ResNet')
