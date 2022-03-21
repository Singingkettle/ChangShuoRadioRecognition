from .alexnetco import AlexNetCo
from ..builder import TASKS


@TASKS.register_module()
class ResNetCo(AlexNetCo):
    def __init__(self, backbone, classifier_head, method_name='ResNetCO', is_iq=True, train_cfg=None, test_cfg=None):
        super(ResNetCo, self).__init__(backbone, classifier_head, method_name, is_iq, train_cfg, test_cfg)

    def init_weights(self, pre_trained=None):
        """Initialize the weights in task.

        Args:
            pre_trained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(ResNetCo, self).init_weights(pre_trained)
        self.backbone.init_weights(pre_trained=pre_trained)

        self.classifier_head.init_weights()
