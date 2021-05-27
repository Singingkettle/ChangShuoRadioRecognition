import torch

from .sei import BaseSEI
from ..builder import TASKS, build_backbone, build_head
from ...common.utils import outs2result


@TASKS.register_module()
class MBRFI(BaseSEI):

    def __init__(self, backbone, classifier_head, train_cfg=None, test_cfg=None, is_dual=False,
                 return_dual_label=False):
        super(MBRFI, self).__init__()
        self.backbone = build_backbone(backbone)
        self.classifier_head = build_head(classifier_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.is_dual = is_dual
        if return_dual_label:
            self.is_dual = True
            self.return_dual_label = True
        else:
            self.return_dual_label = False

        # init weights
        self.init_weights()

    def init_weights(self, pre_trained=None):
        """Initialize the weights in task.

        Args:
            pre_trained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(MBRFI, self).init_weights(pre_trained)
        self.backbone.init_weights(pre_trained=pre_trained)

        self.classifier_head.init_weights()

    def extract_feat(self, *args, **kwargs):
        """Directly extract features from the backbone."""
        x = self.backbone(*args, **kwargs)
        return x

    def forward_train(self, *args, **kwargs):
        if self.is_dual:
            cos = torch.cat([args[0]['cos'], args[1]['cos']], dim=0)
            dev_labels = torch.cat([args[0]['dev_labels'], args[1]['dev_labels']], dim=0)
            x = self.extract_feat(cos)
            if self.return_dual_label:
                losses = self.classifier_head.forward_train(x, dev_labels=dev_labels,
                                                            from_same_dev=args[2]['from_same_dev'])
            else:
                losses = self.classifier_head.forward_train(x, dev_labels=dev_labels)
        else:
            dev_labels = kwargs.pop('dev_labels')
            x = self.extract_feat(**kwargs)
            losses = self.classifier_head.forward_train(x, dev_labels=dev_labels)

        return losses

    def simple_test(self, **kwargs):
        x = self.extract_feat(**kwargs)
        outs = self.classifier_head(x)
        if self.return_dual_label:
            outs = outs['c_x']

        results_list = []
        for idx in range(outs.shape[0]):
            result = outs2result(outs[idx, :])
            results_list.append(result)

        return results_list

    def forward_dummy(self, **kwargs):
        """Used for computing network flops.
        See `wtisignalprocessing/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(**kwargs)
        outs = self.classifier_head(x)
        return outs
