import logging

import torch.nn as nn

from ..builder import BACKBONES
from ...runner import load_checkpoint


@BACKBONES.register_module()
class TanhNet(nn.Module):
    """Backbone with the tanh layer
    """

    def __init__(self, signal_length):
        super(TanhNet, self).__init__()
        self.signal_length = signal_length
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(1, signal_length)),
            nn.Tanh(),
        )

    def init_weights(self, pre_trained=None):
        if isinstance(pre_trained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pre_trained, strict=False, logger=logger)
        elif pre_trained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outs = self.backbone(x)

        return outs
