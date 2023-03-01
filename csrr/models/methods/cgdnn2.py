from .base_classifier import BaseClassifier

from ..builder import METHODS


@METHODS.register_module()
class CGDNN2(BaseClassifier):
    def __init__(self, backbone, classifier_head, vis_fea=False):
        super(CGDNN2, self).__init__(backbone, classifier_head, vis_fea, 'CGDNN2')
