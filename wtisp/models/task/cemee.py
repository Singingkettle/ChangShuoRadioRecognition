import torch

from .amc import BaseAMC
from ..builder import TASKS, build_backbone, build_head
from ...common.utils import outs2result


@TASKS.register_module()
class CEMEE(BaseAMC):

    def __init__(
            self,
            backbone,
            classifier_head,
            channel_mode=False,
            train_cfg=None,
            test_cfg=None,
            pre_trained=None):
        super(CEMEE, self).__init__()
        self.backbone = build_backbone(backbone)
        self.classifier_head = build_head(classifier_head)
        self.channel_mode = channel_mode
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # init weights
        self.init_weights(pre_trained)

    def init_weights(self, pre_trained=None):
        """Initialize the weights in task.

        Args:
            pre_trained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(CEMEE, self).init_weights(pre_trained)
        self.backbone.init_weights(pre_trained=pre_trained)

        self.classifier_head.init_weights()

    def extract_feat(self, x):
        """Directly extract features from the backbone."""
        x = self.backbone(x)
        return x

    def forward_train(
            self,
            iqs,
            aps,
            cos,
            mod_labels,
            hard_labels=None,
            snr_labels=None,
            low_weight=None,
            high_weight=None):
        x = self.extract_feat(iqs)
        losses = self.classifier_head.forward_train(
            x, mod_labels=mod_labels, snr_labels=snr_labels, low_weight=low_weight, high_weight=high_weight)

        return losses

    def simple_test(self, iqs, aps, cos):
        x = self.extract_feat(iqs)
        outs = self.classifier_head(x, mode='test')
        results_list = []
        for idx in range(outs.shape[0]):
            result = outs2result(outs[idx, :])
            results_list.append(result)

        return results_list
