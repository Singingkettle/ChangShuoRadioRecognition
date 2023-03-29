import torch.nn as nn

from ..builder import BACKBONES


@BACKBONES.register_module()
class SSRCNN(nn.Module):
    def __init__(self, num_channels=2):
        super(SSRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=[5, 1], stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=[2, 1], stride=[2, 1], return_indices=True)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=[3, 1], stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=[3, 1], stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.5)
        self.maxUnpool = nn.MaxUnpool2d(kernel_size=[2, 1], stride=[2, 1])

        self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=[3, 1], stride=1, bias=False)
        self.debn3 = nn.BatchNorm2d(32)
        self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=[3, 1], stride=1, bias=False)
        self.debn2 = nn.BatchNorm2d(32)
        self.deconv1 = nn.ConvTranspose2d(32, 1, kernel_size=[5, 1], stride=1, bias=False)
        self.debn1 = nn.BatchNorm2d(1)
