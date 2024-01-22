import torch
import torch.nn as nn

from .base_backbone import BaseBackbone
from ..builder import BACKBONES


class PET(nn.Module):
    def __init__(self, frame_length=128):
        super(PET, self).__init__()
        self.p1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(frame_length*2, 1),
        )

    def forward(self, x):
        p1_x = self.p1(x)
        sin_x = torch.sin(p1_x)
        cos_x = torch.cos(p1_x)

        x11 = x[:, :, 0] * cos_x
        x12 = x[:, :, 1] * sin_x
        x21 = x[:, :, 0] * sin_x
        x22 = x[:, :, 1] * cos_x

        y1 = x11 + x12
        y2 = x21 - x22
        y1 = torch.unsqueeze(y1, 2)
        y2 = torch.unsqueeze(y2, 2)

        x2 = torch.cat([y1, y2], dim=2)
        x2 = torch.transpose(x2, 1, 2)
        x2 = torch.unsqueeze(x2, 1)

        return x2


@BACKBONES.register_module()
class PETCGDNN(BaseBackbone):
    """`PETCGDNN <https://ieeexplore.ieee.org/abstract/document/9507514>`_ backbone
    The input for PETCGDNN is an N*L*2 frame
    Args:
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, num_classes=-1, frame_length=128, hidden_size=128, init_cfg=None):
        super(PETCGDNN, self).__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.features = nn.Sequential(
            PET(frame_length=frame_length),
            nn.Conv2d(1, 75, kernel_size=(2, 8), padding='valid'),
            nn.ReLU(inplace=True),
            nn.Conv2d(75, 25, kernel_size=(1, 5), padding='valid'),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(input_size=25, hidden_size=hidden_size, batch_first=True)
        if self.num_classes > 0:
            self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        x = self.features(x)
        x = torch.squeeze(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.gru(x)
        if self.num_classes > 0:
            x = self.classifier(x[:, -1, :])

        return (x,)
