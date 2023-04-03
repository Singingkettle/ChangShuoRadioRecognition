from .base_classifier import BaseClassifier

from ..builder import METHODS


@METHODS.register_module()
class SSRCNN(BaseClassifier):
    def __init__(self, backbone, classifier_head, vis_fea=False):
        super(SSRCNN, self).__init__(backbone, classifier_head, vis_fea, 'SSRCNN')
