from .base_classifier import BaseClassifier

from ..builder import METHODS


@METHODS.register_module()
class CNN2(BaseClassifier):
    def __init__(self, backbone, classifier_head, vis_fea=False):
        super(CNN2, self).__init__(backbone, classifier_head, vis_fea, 'CNN2')
