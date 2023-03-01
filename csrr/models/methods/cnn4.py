from .base_classifier import BaseClassifier

from ..builder import METHODS


@METHODS.register_module()
class CNN4(BaseClassifier):
    def __init__(self, backbone, classifier_head, vis_fea=False):
        super(CNN4, self).__init__(backbone, classifier_head, vis_fea, 'CNN4')
