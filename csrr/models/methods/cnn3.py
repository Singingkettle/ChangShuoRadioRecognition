from .single_head_classifier import SingleHeadClassifier

from ..builder import TASKS


@TASKS.register_module()
class CNN3(SingleHeadClassifier):
    def __init__(self, backbone, classifier_head, vis_fea=False):
        super(CNN3, self).__init__(backbone, classifier_head, vis_fea, 'CNN3')
