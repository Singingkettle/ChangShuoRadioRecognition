from .base_classifier import BaseClassifier

from ..builder import METHODS


@METHODS.register_module()
class MCformer(BaseClassifier):
    def __init__(self, backbone, classifier_head, method_name, vis_fea=False):
        super(MCformer, self).__init__(backbone, classifier_head, vis_fea, method_name)