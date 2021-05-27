from .amc import BaseAMC
from ..builder import TASKS, build_backbone, build_head
from ...common.utils import outs2result


@TASKS.register_module()
class MLDNN(BaseAMC):

    def __init__(self, backbone, classifier_head, train_cfg=None, test_cfg=None):
        super(MLDNN, self).__init__()
        self.backbone = build_backbone(backbone)
        self.classifier_head = build_head(classifier_head)
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
        super(MLDNN, self).init_weights(pre_trained)
        self.backbone.init_weights(pre_trained=pre_trained)

        self.classifier_head.init_weights()

    def extract_feat(self, x1, x2):
        """Directly extract features from the backbone."""
        x = self.backbone(x1, x2)
        return x

    def forward_train(self, iqs, aps, cos, snr_labels, mod_labels):
        x = self.extract_feat(iqs, aps)
        losses = self.classifier_head.forward_train(
            x, mod_labels=mod_labels, snr_labels=snr_labels)

        return losses

    def simple_test(self, iqs, aps, cos):
        x = self.extract_feat(iqs, aps)
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

    def forward_dummy(self, iqs, aps, cos):
        """Used for computing network flops.
        See `wtisignalprocessing/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(iqs, aps)
        outs = self.classifier_head(x)
        return outs
