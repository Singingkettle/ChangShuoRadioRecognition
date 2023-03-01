from .base import BaseDNN
from ..builder import METHODS, build_backbone, build_head
from ...common.utils import outs2result


@METHODS.register_module()
class BaseClassifier(BaseDNN):

    def __init__(self, backbone, classifier_head, vis_fea=False, method_name=None):
        super(BaseClassifier, self).__init__()
        self.backbone = build_backbone(backbone)
        self.classifier_head = build_head(classifier_head)
        self.vis_fea = vis_fea
        if method_name is None:
            raise ValueError('You should give a method name when using this method class!')
        else:
            self.method_name = method_name

        # init weights
        self.init_weights()

    def init_weights(self, pre_trained=None):
        """Initialize the weights in method.

        Args:
            pre_trained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(BaseClassifier, self).init_weights(pre_trained)
        self.backbone.init_weights(pre_trained=pre_trained)

        self.classifier_head.init_weights()

    def extract_feat(self, inputs):
        """Directly extract features from the backbone."""
        x = self.backbone(**inputs)
        return x

    def forward_train(self, inputs, input_metas, targets, **kwargs):
        x = self.extract_feat(inputs)
        losses = self.classifier_head.forward_train(x, targets, **kwargs)

        return losses

    def simple_test(self, inputs, input_metas, **kwargs):
        x = self.extract_feat(inputs)
        outs = self.classifier_head(x, self.vis_fea, True)

        results_list = []
        if isinstance(outs, dict):
            keys = list(outs.keys())
            batch_size = outs[keys[0]].shape[0]
            for idx in range(batch_size):
                item = dict()
                for key_str in keys:
                    item[key_str] = outs2result(outs[key_str][idx, :])
                results_list.append(item)
            return results_list
        else:
            for idx in range(outs.shape[0]):
                result = outs2result(outs[idx, :])
                results_list.append(result)
            return results_list

    def forward_dummy(self, inputs):
        x = self.extract_feat(inputs)
        outs = self.classifier_head(x)
        return outs
