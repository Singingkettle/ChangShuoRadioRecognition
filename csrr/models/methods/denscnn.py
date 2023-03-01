from .base_classifier import BaseClassifier

from ..builder import METHODS


@METHODS.register_module()
class DensCNN(BaseClassifier):
    def __init__(self, backbone, classifier_head, vis_fea=False):
        super(DensCNN, self).__init__(backbone, classifier_head, vis_fea, 'DensCNN')
