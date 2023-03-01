from .base_classifier import BaseClassifier

from ..builder import METHODS


@METHODS.register_module()
class CNN3(BaseClassifier):
    def __init__(self, backbone, classifier_head, vis_fea=False):
        super(CNN3, self).__init__(backbone, classifier_head, vis_fea, 'CNN3')
