from csrr.models.backbone.base import BaseBackbone
import torch.nn as nn
from torch_receptive_field import receptive_field
from torchscan import summary


class DetCNN(BaseBackbone):

    def __init__(self, init_cfg=None):
        super(DetCNN, self).__init__(init_cfg)
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=6, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 16, kernel_size=5, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 16, kernel_size=4, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
        )

    def forward(self, iqs):
        x1 = self.conv1(iqs)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        return x5

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = DetCNN().to(device)

receptive_field_dict = receptive_field(model, (1, 240, 240))