from .base import BaseDNN
from ..builder import TASKS, build_backbone, build_head
from ...common.utils import outs2result


@TASKS.register_module()
class DNN(BaseDNN):

    def __init__(self, backbone, classifier_head, vis_fea=False, method_name=None):
        super(DNN, self).__init__()
        self.backbone = build_backbone(backbone)
        self.classifier_head = build_head(classifier_head)
        self.vis_fea = vis_fea
        if method_name is None:
            raise ValueError('You should give a method name when using this task class!')
        else:
            self.method_name = method_name

        # init weights
        self.init_weights()

    def init_weights(self, pre_trained=None):
        """Initialize the weights in task.

        Args:
            pre_trained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(DNN, self).init_weights(pre_trained)
        self.backbone.init_weights(pre_trained=pre_trained)

        self.classifier_head.init_weights()

    def extract_feat(self, **kwargs):
        """Directly extract features from the backbone."""
        x = self.backbone(**kwargs)
        return x

    def forward_train(self, **kwargs):
        for_train = dict()
        for key_str in list(kwargs.keys()):
            if 'labels' in key_str or 'weight' in key_str:
                for_train[key_str] = kwargs[key_str]
                kwargs.pop(key_str, None)
        x = self.extract_feat(**kwargs)
        losses = self.classifier_head.forward_train(x, **for_train)

        return losses

    def simple_test(self, **kwargs):
        x = self.extract_feat(**kwargs)
        outs = self.classifier_head(x, self.vis_fea, True)

        results_list = []
        if isinstance(outs, dict):
            keys = list(outs.keys())
            batch_size = outs[keys[0]].shape[0]
            for idx in range(batch_size):
                item = dict()
                for key_str in keys:
                    if key_str != 'Final' or 'fea' in key_str:
                        save_name = self.method_name + '-' + key_str
                    else:
                        save_name = 'Final'
                    item[save_name] = outs2result(outs[key_str][idx, :])
                results_list.append(item)
        else:
            for idx in range(outs.shape[0]):
                result = outs2result(outs[idx, :])
                result = {'Final': result}
                results_list.append(result)
        return results_list

    def forward_dummy(self, **kwargs):
        x = self.extract_feat(**kwargs)
        outs = self.classifier_head(x)
        return outs
