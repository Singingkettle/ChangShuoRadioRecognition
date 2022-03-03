from .amc import BaseAMC
from ..builder import TASKS, build_backbone, build_head
from ...common.utils import outs2result


@TASKS.register_module()
class SCDN(BaseAMC):

    def __init__(self, backbone, classifier_head, is_iq=True, train_cfg=None, test_cfg=None):
        super(SCDN, self).__init__()
        self.backbone = build_backbone(backbone)  # self.backbone = SCDN()
        # self.classifier_head = AMCHead()
        self.classifier_head = build_head(classifier_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.is_iq = is_iq

        # init weights
        self.init_weights()

    def init_weights(self, pre_trained=None):
        """Initialize the weights in task.

        Args:
            pre_trained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(SCDN, self).init_weights(pre_trained)
        self.backbone.init_weights(pre_trained=pre_trained)

        self.classifier_head.init_weights()

    def extract_feat(self, x):
        """Directly extract features from the backbone."""
        x = self.backbone(x)
        return x

    def forward_train(self, iqs, aps, cos, mod_labels):
        x = self.extract_feat(cos)

        out = self.classifier_head(x)
        loss = self.classifier_head.loss(x, mod_labels)

        losses = self.classifier_head.forward_train(x, mod_labels=mod_labels)

        return losses

    def simple_test(self, iqs, aps, cos):
        x = self.extract_feat(cos)
        outs = self.classifier_head(x)  # self.classifier_head.forward(x)

        results_list = []
        for idx in range(outs.shape[0]):
            result = outs2result(outs[idx, :])
            results_list.append(result)

        return results_list
