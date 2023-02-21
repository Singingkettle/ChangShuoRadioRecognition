from .single_head_classifier import SingleHeadClassifier

from ..builder import TASKS


@TASKS.register_module()
class DensCNN(SingleHeadClassifier):
    def __init__(self, backbone, classifier_head, vis_fea=False):
        super(DensCNN, self).__init__(backbone, classifier_head, vis_fea, 'DensCNN')
