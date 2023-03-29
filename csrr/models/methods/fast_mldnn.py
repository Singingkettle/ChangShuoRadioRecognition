from .base_classifier import BaseClassifier

from ..builder import METHODS
from ...common.utils import outs2result


@METHODS.register_module()
class FastMLDNN(BaseClassifier):
    def __init__(self, backbone, classifier_head, vis_fea=False, init_cfg=None):
        super(FastMLDNN, self).__init__(backbone, classifier_head, vis_fea, 'FastMLDNN', init_cfg)

    def forward_test(self, inputs, **kwargs):
        x = self.extract_feat(inputs)
        outs = self.classifier_head(x, self.vis_fea, True)

        results_list = []
        if isinstance(outs, dict):
            center = outs2result(outs['center'])
            outs.pop('center')
            keys = list(outs.keys())
            batch_size = outs[keys[0]].shape[0]
            for idx in range(batch_size):
                item = dict()
                for key_str in keys:
                    item[key_str] = outs2result(outs[key_str][idx, :])
                item['center'] = center
                results_list.append(item)
            return results_list
        else:
            for idx in range(outs.shape[0]):
                result = outs2result(outs[idx, :])
                results_list.append(result)
            return results_list
