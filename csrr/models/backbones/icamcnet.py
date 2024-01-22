import torch
import torch.nn as nn

from .base_backbone import BaseBackbone
from ..builder import BACKBONES


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value you are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.expand(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x


@BACKBONES.register_module()
class ICAMCNet(BaseBackbone):
    """`ICAMCNet <https://ieeexplore.ieee.org/abstract/document/8977561>`_ backbone
    The input for ICAMCNet is a 1*2*L frame
    Args:
        frame_length (int): the frame length equal to number of sample points
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, frame_length=128, num_classes=-1, init_cfg=None):
        super(ICAMCNet, self).__init__(init_cfg=init_cfg)
        self.frame_length = frame_length
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 8), padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(64, 64, kernel_size=(1, 4), padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(1, 8), padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 1), stride=1),
            nn.Dropout(0.4),
            nn.Conv2d(128, 128, kernel_size=(1, 8), padding='same'),
            nn.Dropout(0.4),
            nn.Flatten(),
        )
        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(self.frame_length // 2 * 128, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                GaussianNoise(1),
                nn.Linear(128, self.num_classes),
            )

    def forward(self, x):
        x = self.features(x)
        if self.num_classes > 0:
            x = self.classifier(x)

        return (x,)
