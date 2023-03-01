from .base_classifier import BaseClassifier

from ..builder import METHODS


@METHODS.register_module()
class CLDNN(BaseClassifier):
    def __init__(self, backbone, classifier_head, vis_fea=False):
        super(CLDNN, self).__init__(backbone, classifier_head, vis_fea, 'CLDNN')
