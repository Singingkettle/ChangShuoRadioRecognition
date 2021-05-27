from .separator import BaseSeparator
from ..builder import TASKS, build_backbone, build_head
from ...common.utils import outs2result


@TASKS.register_module()
class TCNN(BaseSeparator):

    def __init__(self, backbone, filter_head, train_cfg=None, test_cfg=None):
        super(TCNN, self).__init__()
        self.backbone = build_backbone(backbone)
        self.filter_head = build_head(filter_head)
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
        super(TCNN, self).init_weights(pre_trained)
        self.backbone.init_weights(pre_trained=pre_trained)

        self.filter_head.init_weights()

    def extract_feat(self, mix):
        """Directly extract features from the backbone."""
        x = self.backbone(mix)
        return x

    def forward_train(self, mix, target):
        x = self.extract_feat(mix)
        losses = self.filter_head.forward_train(x, target)

        return losses

    def simple_test(self, mix):
        x = self.extract_feat(mix)
        outs = self.filter_head(x)

        results = outs2result(outs)

        return results
