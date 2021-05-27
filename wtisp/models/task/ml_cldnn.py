from .amc import BaseAMC
from ..builder import TASKS, build_backbone, build_head
from ...common.utils import outs2result


@TASKS.register_module()
class MLCLDNN(BaseAMC):

    def __init__(self, backbone, classifier_head, is_iq=True, train_cfg=None, test_cfg=None):
        super(MLCLDNN, self).__init__()
        self.backbone = build_backbone(backbone)
        self.classifier_head = build_head(classifier_head)
        self.is_iq = is_iq
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # init weights
        self.init_weights()

    def init_weights(self, pre_trained=None):
        """Initialize the weights in task.

        Args:
            pre_trained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(MLCLDNN, self).init_weights(pre_trained)
        self.backbone.init_weights(pre_trained=pre_trained)

        self.classifier_head.init_weights()

    def extract_feat(self, x):
        """Directly extract features from the backbone."""
        x = self.backbone(x)
        return x

    def forward_train(self, iqs, aps, cos, mod_labels, snr_labels, low_weight, high_weight):
        if self.is_iq:
            x = self.extract_feat(iqs)
        else:
            x = self.extract_feat(aps)
        losses = self.classifier_head.forward_train(
            x, mod_labels=mod_labels, snr_labels=snr_labels, low_weight=low_weight, high_weight=high_weight)

        return losses

    def simple_test(self, iqs, aps, cos):
        if self.is_iq:
            x = self.extract_feat(iqs)
        else:
            x = self.extract_feat(aps)
        outs = self.classifier_head(x)

        results_list = []
        keys = list(outs.keys())
        batch_size = outs[keys[0]].shape[0]
        for idx in range(batch_size):
            item = dict()
            for key_str in keys:
                item[key_str] = outs2result(outs[key_str][idx, :])
            results_list.append(item)

        return results_list
