import torch
from torch import nn

from .base_backbone import BaseBackbone
from ..builder import BACKBONES


@BACKBONES.register_module()
class MCformer(BaseBackbone):
    """`MCformer <https://ieeexplore.ieee.org/abstract/document/9685815>`_ backbone
    The input for MCformer is a 1*2*L frame
    Args:
        frame_length (int): the frame length equal to number of sample points
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """
    def __init__(self, fea_dim=32, frame_length=128, num_classes=-1, init_cfg=None):
        super(MCformer, self).__init__(init_cfg)
        self.frame_length = frame_length
        self.num_classes = num_classes
        self.cnn = nn.Sequential(
            nn.Conv1d(2, fea_dim, kernel_size=65, padding='same'),
            nn.ReLU(inplace=True),
        )
        self.fea_dim = fea_dim
        encoder_layer = nn.TransformerEncoderLayer(fea_dim, 4, dim_feedforward=fea_dim, batch_first=True)
        self.tnn = nn.TransformerEncoder(encoder_layer, num_layers=4)

        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(4 * self.fea_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, self.num_classes)
            )

    def forward(self, x):
        x = self.cnn(x)
        x = torch.squeeze(x, dim=2)
        x = torch.transpose(x, 1, 2)
        x = self.tnn(x)
        x = x[:, :4, :]
        x = torch.reshape(x, [-1, 4 * self.fea_dim])
        if self.num_classes > 0:
            x = self.classifier(x)

        return (x,)
