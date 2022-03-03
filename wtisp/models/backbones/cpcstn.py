import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import BACKBONES
from ...runner import load_checkpoint


@BACKBONES.register_module()
class CPCNN(nn.Module):
    def __init__(self, num_slot, num_filter, slot_size):
        super(CPCNN, self).__init__()
        self.slot_size = slot_size
        self.num_slot = num_slot
        self.fea_num = int(num_filter * slot_size * slot_size / 64)

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, num_filter, kernel_size=(
                3, 3), stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_filter, 2 * num_filter, kernel_size=(3, 3),
                      stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv3 = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(2 * num_filter, num_filter,
                      kernel_size=(2, 2), stride=1, padding=0),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))

        self.rnn_layer1 = nn.LSTM(self.fea_num, 64, batch_first=True)
        self.rnn_layer2 = nn.LSTM(self.fea_num, 64, batch_first=True)
        self.den = nn.Sequential(
            nn.Linear(64 * num_slot, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
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

    def forward(self, co):
        # co (80, 9, 128, 128)
        # co = self.stn_layer(co)
        x = co.view(-1, 1, self.slot_size, self.slot_size)  # co (80*9, 1, 128, 128)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)  # (80*9, 3, 16, 16)
        fea = x3.view(-1, 1, self.fea_num)  # (80*9, 1, 3*16*16)

        out = fea.view(-1, self.num_slot, self.fea_num)  # (80, 9, 3*16*16)
        out1, _ = self.rnn_layer1(out)  #
        out2, _ = self.rnn_layer2(out)  # (N, L, 64)
        out2 = torch.flip(out2, [1])  # reverse
        out = torch.add(out1, out2)
        out = torch.flatten(out, start_dim=1)
        out = self.den(out)  # self.den.forward(out)

        return out


class STNNet(nn.Module):
    def __init__(self, slot_size):
        super(STNNet, self).__init__()
        self.in_fea = 10 * (((((slot_size - 7) + 1) // 2 - 5) + 1) // 2) ** 2
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.in_fea, 32),
            nn.ReLU(True),
            # nn.Linear(32, 2 * 2)
            nn.Linear(32, 1 * 1)

        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        # self.fc_loc[2].bias.data.copy_(torch.tensor(
        #     [1, 0, 0, 1], dtype=torch.float))
        self.fc_loc[2].bias.data.copy_(torch.tensor(
            [0], dtype=torch.float))

    # Spatial transformer network forward function
    def forward(self, x):  # (80*9,1,128,128)
        xs = self.localization(x)  # (80*9,10,28,28)
        xs = xs.view(-1, self.in_fea)  # (80*9,7840)
        theta = self.fc_loc(xs)  # (90,4)
        theta = torch.tanh(theta) * (math.pi)  # start
        theta1 = []
        theta2 = []
        theta1 = torch.cos(theta)  # (90, 1)
        theta2 = torch.sin(theta)  # (90, 1)
        theta = torch.stack([theta1, -theta2, theta2, theta1], 2)  # (90, 4) #end
        theta = theta.view(-1, 2, 2)  # (90*8,2,2)
        p1d = (0, 1)
        theta = F.pad(theta, p1d, "constant", 0)  # (90*8,2,3)

        grid = F.affine_grid(theta, x.size())  # x.size(80*9, 1, 128, 128)   grid(80, 128, 128, 2)
        x = F.grid_sample(x, grid)  # 80*9, 1, 128, 128

        return x


class STNNet1(nn.Module):
    def __init__(self, slot_size):
        super(STNNet1, self).__init__()
        self.in_fea = 10 * (((((slot_size - 5) + 1) // 2 - 3) + 1) // 2) ** 2
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.in_fea, 32),
            nn.ReLU(True),
            # nn.Linear(32, 2 * 2)
            nn.Linear(32, 1 * 1)

        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        # self.fc_loc[2].bias.data.copy_(torch.tensor(
        #     [1, 0, 0, 1], dtype=torch.float))
        self.fc_loc[2].bias.data.copy_(torch.tensor(
            [0], dtype=torch.float))

    # Spatial transformer network forward function
    def forward(self, x):  # (80*9,1,128,128)
        xs = self.localization(x)  # (80*9,10,28,28)
        xs = xs.view(-1, self.in_fea)  # (80*9,7840)
        theta = self.fc_loc(xs)  # (90,4)
        theta = torch.tanh(theta) * (math.pi)  # start
        theta1 = []
        theta2 = []
        theta1 = torch.cos(theta)  # (90, 1)
        theta2 = torch.sin(theta)  # (90, 1)
        theta = torch.stack([theta1, -theta2, theta2, theta1], 2)  # (90, 4) #end
        theta = theta.view(-1, 2, 2)  # (90*8,2,2)
        p1d = (0, 1)
        theta = F.pad(theta, p1d, "constant", 0)  # (90*8,2,3)

        grid = F.affine_grid(theta, x.size())  # x.size(80*9, 1, 128, 128)   grid(80, 128, 128, 2)
        x = F.grid_sample(x, grid)  # 80*9, 1, 128, 128

        return x


class STNNet2(nn.Module):
    def __init__(self, slot_size):
        super(STNNet2, self).__init__()
        self.in_fea = 10 * (((((slot_size - 3) + 1) // 2 - 1) + 1) // 2) ** 2
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.in_fea, 32),
            nn.ReLU(True),
            # nn.Linear(32, 2 * 2)
            nn.Linear(32, 1 * 1)

        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        # self.fc_loc[2].bias.data.copy_(torch.tensor(
        #     [1, 0, 0, 1], dtype=torch.float))
        self.fc_loc[2].bias.data.copy_(torch.tensor(
            [0], dtype=torch.float))

    # Spatial transformer network forward function
    def forward(self, x):  # (80*9,1,128,128)
        xs = self.localization(x)  # (80*9,10,28,28)
        xs = xs.view(-1, self.in_fea)  # (80*9,7840)
        theta = self.fc_loc(xs)  # (90,4)
        theta = torch.tanh(theta) * (math.pi)  # start
        theta1 = []
        theta2 = []
        theta1 = torch.cos(theta)  # (90, 1)
        theta2 = torch.sin(theta)  # (90, 1)
        theta = torch.stack([theta1, -theta2, theta2, theta1], 2)  # (90, 4) #end
        theta = theta.view(-1, 2, 2)  # (90*8,2,2)
        p1d = (0, 1)
        theta = F.pad(theta, p1d, "constant", 0)  # (90*8,2,3)

        grid = F.affine_grid(theta, x.size())  # x.size(80*9, 1, 128, 128)   grid(80, 128, 128, 2)
        x = F.grid_sample(x, grid)  # 80*9, 1, 128, 128

        return x


class STNNet3(nn.Module):
    def __init__(self, slot_size):
        super(STNNet3, self).__init__()
        # self.in_fea = 10 * (((((slot_size - 7) + 1) // 2 - 5) + 1) // 2) ** 2
        self.in_fea = 10 * (((((slot_size - 9) + 1) // 2 - 7) + 1) // 2) ** 2
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=9),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.in_fea, 32),
            nn.ReLU(True),
            # nn.Linear(32, 2 * 2)
            nn.Linear(32, 1 * 1)

        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        # self.fc_loc[2].bias.data.copy_(torch.tensor(
        #     [1, 0, 0, 1], dtype=torch.float))
        self.fc_loc[2].bias.data.copy_(torch.tensor(
            [0], dtype=torch.float))

    # Spatial transformer network forward function
    def forward(self, x):  # (80*9,1,128,128)
        xs = self.localization(x)  # (80*9,10,28,28)
        xs = xs.view(-1, self.in_fea)  # (80*9,7840)
        theta = self.fc_loc(xs)  # (90,4)
        theta = torch.tanh(theta) * (math.pi)  # start
        theta1 = []
        theta2 = []
        theta1 = torch.cos(theta)  # (90, 1)
        theta2 = torch.sin(theta)  # (90, 1)
        theta = torch.stack([theta1, -theta2, theta2, theta1], 2)  # (90, 4) #end
        theta = theta.view(-1, 2, 2)  # (90*8,2,2)
        p1d = (0, 1)
        theta = F.pad(theta, p1d, "constant", 0)  # (90*8,2,3)

        grid = F.affine_grid(theta, x.size())  # x.size(80*9, 1, 128, 128)   grid(80, 128, 128, 2)
        x = F.grid_sample(x, grid)  # 80*9, 1, 128, 128

        return x


class STNNet4(nn.Module):
    def __init__(self, slot_size):
        super(STNNet4, self).__init__()
        # self.in_fea = 10 * (((((slot_size - 7) + 1) // 2 - 5) + 1) // 2) ** 2
        self.in_fea = 10 * (((((slot_size - 11) + 1) // 2 - 9) + 1) // 2) ** 2
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=11),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=9),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.in_fea, 32),
            nn.ReLU(True),
            # nn.Linear(32, 2 * 2)
            nn.Linear(32, 1 * 1)

        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        # self.fc_loc[2].bias.data.copy_(torch.tensor(
        #     [1, 0, 0, 1], dtype=torch.float))
        self.fc_loc[2].bias.data.copy_(torch.tensor(
            [0], dtype=torch.float))

    # Spatial transformer network forward function
    def forward(self, x):  # (80*9,1,128,128)
        xs = self.localization(x)  # (80*9,10,28,28)
        xs = xs.view(-1, self.in_fea)  # (80*9,7840)
        theta = self.fc_loc(xs)  # (90,4)
        theta = torch.tanh(theta) * (math.pi)  # start
        theta1 = []
        theta2 = []
        theta1 = torch.cos(theta)  # (90, 1)
        theta2 = torch.sin(theta)  # (90, 1)
        theta = torch.stack([theta1, -theta2, theta2, theta1], 2)  # (90, 4) #end
        theta = theta.view(-1, 2, 2)  # (90*8,2,2)
        p1d = (0, 1)
        theta = F.pad(theta, p1d, "constant", 0)  # (90*8,2,3)

        grid = F.affine_grid(theta, x.size())  # x.size(80*9, 1, 128, 128)   grid(80, 128, 128, 2)
        x = F.grid_sample(x, grid)  # 80*9, 1, 128, 128

        return x


class SlotbasicBlock(nn.Module):
    def __init__(self, num_filter):
        super(SlotbasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, num_filter, kernel_size=(
                3, 3), stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_filter, 2 * num_filter, kernel_size=(3, 3),
                      stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv3 = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(2 * num_filter, num_filter,
                      kernel_size=(2, 2), stride=1, padding=0),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))

    def forward(self, co):
        x1 = self.conv1(co)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        return x3


class STNNet5(nn.Module):
    def __init__(self, slot_size):
        super(STNNet5, self).__init__()
        self.in_fea = 10 * (((((((slot_size - 5) + 1) // 2 - 3) + 1) // 2 - 3) + 1) // 2) ** 2
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(5, 8, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.in_fea, 32),
            nn.ReLU(True),
            # nn.Linear(32, 2 * 2)
            nn.Linear(32, 1 * 1)

        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        # self.fc_loc[2].bias.data.copy_(torch.tensor(
        #     [1, 0, 0, 1], dtype=torch.float))
        self.fc_loc[2].bias.data.copy_(torch.tensor(
            [0], dtype=torch.float))

    # Spatial transformer network forward function
    def forward(self, x):  # (80*9,1,128,128)
        xs = self.localization(x)  # (80*9,10,14,14)
        xs = xs.view(-1, self.in_fea)  # (80*9,1960)
        theta = self.fc_loc(xs)  # (90,1)
        theta = torch.tanh(theta) * (math.pi)  # start
        theta1 = []
        theta2 = []
        theta1 = torch.cos(theta)  # (90, 1)
        theta2 = torch.sin(theta)  # (90, 1)
        theta = torch.stack([theta1, -theta2, theta2, theta1], 2)  # (90, 4) #end
        theta = theta.view(-1, 2, 2)  # (90*8,2,2)
        p1d = (0, 1)
        theta = F.pad(theta, p1d, "constant", 0)  # (90*8,2,3)

        grid = F.affine_grid(theta, x.size())  # x.size(80*9, 1, 128, 128)   grid(80, 128, 128, 2)
        x = F.grid_sample(x, grid)  # 80*9, 1, 128, 128

        return x


class STNNet6(nn.Module):
    def __init__(self, slot_size):
        super(STNNet6, self).__init__()
        self.in_fea = 10 * (((((((slot_size - 5) + 1) // 2 - 3) + 1) // 2 - 3) + 1) // 2) ** 2
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(3, 5, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(5, 10, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.in_fea, 32),
            nn.ReLU(True),
            # nn.Linear(32, 2 * 2)
            nn.Linear(32, 1 * 1)

        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        # self.fc_loc[2].bias.data.copy_(torch.tensor(
        #     [1, 0, 0, 1], dtype=torch.float))
        self.fc_loc[2].bias.data.copy_(torch.tensor(
            [0], dtype=torch.float))

    # Spatial transformer network forward function
    def forward(self, x):  # (80*9,1,128,128)
        xs = self.localization(x)  # (80*9,10,28,28)
        xs = xs.view(-1, self.in_fea)  # (80*9,7840)
        theta = self.fc_loc(xs)  # (90,4)
        theta = torch.tanh(theta) * (math.pi)  # start
        theta1 = []
        theta2 = []
        theta1 = torch.cos(theta)  # (90, 1)
        theta2 = torch.sin(theta)  # (90, 1)
        theta = torch.stack([theta1, -theta2, theta2, theta1], 2)  # (90, 4) #end
        theta = theta.view(-1, 2, 2)  # (90*8,2,2)
        p1d = (0, 1)
        theta = F.pad(theta, p1d, "constant", 0)  # (90*8,2,3)

        grid = F.affine_grid(theta, x.size())  # x.size(80*9, 1, 128, 128)   grid(80, 128, 128, 2)
        x = F.grid_sample(x, grid)  # 80*9, 1, 128, 128

        return x


class STNNet7(nn.Module):
    def __init__(self, slot_size):
        super(STNNet7, self).__init__()
        self.in_fea = 10 * (((((((slot_size - 5) + 1) // 2 - 3) + 1) // 2 - 3) + 1) // 2) ** 2
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(3, 8, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.in_fea, 32),
            nn.ReLU(True),
            # nn.Linear(32, 2 * 2)
            nn.Linear(32, 1 * 1)

        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        # self.fc_loc[2].bias.data.copy_(torch.tensor(
        #     [1, 0, 0, 1], dtype=torch.float))
        self.fc_loc[2].bias.data.copy_(torch.tensor(
            [0], dtype=torch.float))

    # Spatial transformer network forward function
    def forward(self, x):  # (80*9,1,128,128)
        xs = self.localization(x)  # (80*9,10,28,28)
        xs = xs.view(-1, self.in_fea)  # (80*9,7840)
        theta = self.fc_loc(xs)  # (90,4)
        theta = torch.tanh(theta) * (math.pi)  # start
        theta1 = []
        theta2 = []
        theta1 = torch.cos(theta)  # (90, 1)
        theta2 = torch.sin(theta)  # (90, 1)
        theta = torch.stack([theta1, -theta2, theta2, theta1], 2)  # (90, 4) #end
        theta = theta.view(-1, 2, 2)  # (90*8,2,2)
        p1d = (0, 1)
        theta = F.pad(theta, p1d, "constant", 0)  # (90*8,2,3)

        grid = F.affine_grid(theta, x.size())  # x.size(80*9, 1, 128, 128)   grid(80, 128, 128, 2)
        x = F.grid_sample(x, grid)  # 80*9, 1, 128, 128

        return x


class STNNet8(nn.Module):
    def __init__(self, slot_size):
        super(STNNet8, self).__init__()
        self.in_fea = 10 * (((((((slot_size - 5) + 1) // 2 - 5) + 1) // 2 - 3) + 1) // 2) ** 2
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(5, 8, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.in_fea, 32),
            nn.ReLU(True),
            # nn.Linear(32, 2 * 2)
            nn.Linear(32, 1 * 1)

        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        # self.fc_loc[2].bias.data.copy_(torch.tensor(
        #     [1, 0, 0, 1], dtype=torch.float))
        self.fc_loc[2].bias.data.copy_(torch.tensor(
            [0], dtype=torch.float))

    # Spatial transformer network forward function
    def forward(self, x):  # (80*9,1,128,128)
        xs = self.localization(x)  # (80*9,10,28,28)
        xs = xs.view(-1, self.in_fea)  # (80*9,7840)
        theta = self.fc_loc(xs)  # (90,4)
        theta = torch.tanh(theta) * (math.pi)  # start
        theta1 = []
        theta2 = []
        theta1 = torch.cos(theta)  # (90, 1)
        theta2 = torch.sin(theta)  # (90, 1)
        theta = torch.stack([theta1, -theta2, theta2, theta1], 2)  # (90, 4) #end
        theta = theta.view(-1, 2, 2)  # (90*8,2,2)
        p1d = (0, 1)
        theta = F.pad(theta, p1d, "constant", 0)  # (90*8,2,3)

        grid = F.affine_grid(theta, x.size())  # x.size(80*9, 1, 128, 128)   grid(80, 128, 128, 2)
        x = F.grid_sample(x, grid)  # 80*9, 1, 128, 128

        return x


class STNNet9(nn.Module):
    def __init__(self, slot_size):
        super(STNNet6, self).__init__()
        self.in_fea = 10 * (((((((slot_size - 5) + 1) // 2 - 5) + 1) // 2 - 3) + 1) // 2) ** 2
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(3, 5, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(5, 10, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.in_fea, 32),
            nn.ReLU(True),
            # nn.Linear(32, 2 * 2)
            nn.Linear(32, 1 * 1)

        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        # self.fc_loc[2].bias.data.copy_(torch.tensor(
        #     [1, 0, 0, 1], dtype=torch.float))
        self.fc_loc[2].bias.data.copy_(torch.tensor(
            [0], dtype=torch.float))

    # Spatial transformer network forward function
    def forward(self, x):  # (80*9,1,128,128)
        xs = self.localization(x)  # (80*9,10,28,28)
        xs = xs.view(-1, self.in_fea)  # (80*9,7840)
        theta = self.fc_loc(xs)  # (90,4)
        theta = torch.tanh(theta) * (math.pi)  # start
        theta1 = []
        theta2 = []
        theta1 = torch.cos(theta)  # (90, 1)
        theta2 = torch.sin(theta)  # (90, 1)
        theta = torch.stack([theta1, -theta2, theta2, theta1], 2)  # (90, 4) #end
        theta = theta.view(-1, 2, 2)  # (90*8,2,2)
        p1d = (0, 1)
        theta = F.pad(theta, p1d, "constant", 0)  # (90*8,2,3)

        grid = F.affine_grid(theta, x.size())  # x.size(80*9, 1, 128, 128)   grid(80, 128, 128, 2)
        x = F.grid_sample(x, grid)  # 80*9, 1, 128, 128

        return x


class STNNet10(nn.Module):
    def __init__(self, slot_size):
        super(STNNet7, self).__init__()
        self.in_fea = 10 * (((((((slot_size - 5) + 1) // 2 - 5) + 1) // 2 - 3) + 1) // 2) ** 2
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(3, 8, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.in_fea, 32),
            nn.ReLU(True),
            # nn.Linear(32, 2 * 2)
            nn.Linear(32, 1 * 1)

        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        # self.fc_loc[2].bias.data.copy_(torch.tensor(
        #     [1, 0, 0, 1], dtype=torch.float))
        self.fc_loc[2].bias.data.copy_(torch.tensor(
            [0], dtype=torch.float))

    # Spatial transformer network forward function
    def forward(self, x):  # (80*9,1,128,128)
        xs = self.localization(x)  # (80*9,10,28,28)
        xs = xs.view(-1, self.in_fea)  # (80*9,7840)
        theta = self.fc_loc(xs)  # (90,4)
        theta = torch.tanh(theta) * (math.pi)  # start
        theta1 = []
        theta2 = []
        theta1 = torch.cos(theta)  # (90, 1)
        theta2 = torch.sin(theta)  # (90, 1)
        theta = torch.stack([theta1, -theta2, theta2, theta1], 2)  # (90, 4) #end
        theta = theta.view(-1, 2, 2)  # (90*8,2,2)
        p1d = (0, 1)
        theta = F.pad(theta, p1d, "constant", 0)  # (90*8,2,3)

        grid = F.affine_grid(theta, x.size())  # x.size(80*9, 1, 128, 128)   grid(80, 128, 128, 2)
        x = F.grid_sample(x, grid)  # 80*9, 1, 128, 128

        return x


_STNNET = {'v0': STNNet, 'v1': STNNet1, 'v2': STNNet2, 'v3': STNNet3, 'v4': STNNet4, 'v5': STNNet5,
           'v6': STNNet1, 'v7': STNNet7, 'v8': STNNet8, 'v9': STNNet9, 'v10': STNNet10}


@BACKBONES.register_module()
class CSTN(nn.Module):
    def __init__(self, num_slot, num_filter, slot_size, stn_net_name='v0'):
        super(CSTN, self).__init__()
        self.slotlayers = []
        self.stn_layer = _STNNET[stn_net_name](slot_size)
        self.slot_size = slot_size
        self.num_slot = num_slot
        self.fea_num = int(num_filter * slot_size * slot_size / 64)
        for i in range(num_slot):
            Slot_layers = self.make_Slot_layers(num_filter)
            layers_name = "{} {}".format("layer", i + 1)
            self.add_module(layers_name, Slot_layers)
            self.slotlayers.append(layers_name)
        self.rnn_layer1 = nn.LSTM(self.fea_num, 64, batch_first=True)
        self.rnn_layer2 = nn.LSTM(self.fea_num, 64, batch_first=True)
        self.den = nn.Sequential(
            nn.Linear(64 * num_slot, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
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

    def make_Slot_layers(self, num_filter):
        return SlotbasicBlock(num_filter)

    def forward(self, co):
        # co (80, 9, 128, 128)
        co = co.view(-1, 1, self.slot_size, self.slot_size)  # co (80*9, 1, 128, 128)
        co = self.stn_layer(co)
        co = co.view(-1, self.num_slot, self.slot_size, self.slot_size)
        out = []
        for i, layers_name in enumerate(self.slotlayers):
            layer = getattr(self, layers_name)
            x = co[:, i, :, :]
            x = torch.unsqueeze(x, 1)
            fea = layer(x)  # (80, 3, 16, 16)
            # (N, 1, fea_num)
            fea = fea.view(-1, 1, self.fea_num)  # (80, 1, 3*16*16)
            out.append(fea)

        out = torch.cat(out, 1)  # (N, L, fea_num) (80, 9, 3*16*16)
        out1, _ = self.rnn_layer1(out)  #
        out2, _ = self.rnn_layer2(out)  # (N, L, 64)
        out2 = torch.flip(out2, [1])  # reverse
        out = torch.add(out1, out2)
        out = torch.flatten(out, start_dim=1)
        out = self.den(out)  # self.den.forward(out)

        return out


@BACKBONES.register_module()
class CPCSTN(nn.Module):
    def __init__(self, num_slot, num_filter, slot_size, stn_net_name='v0'):
        super(CPCSTN, self).__init__()
        self.stn_layer = _STNNET[stn_net_name](slot_size)
        # self.stn_layer = STNNet(slot_size)
        self.slot_size = slot_size
        self.num_slot = num_slot
        self.fea_num = int(num_filter * slot_size * slot_size / 64)

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, num_filter, kernel_size=(
                3, 3), stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_filter, 2 * num_filter, kernel_size=(3, 3),
                      stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv3 = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(2 * num_filter, num_filter,
                      kernel_size=(2, 2), stride=1, padding=0),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))

        self.rnn_layer1 = nn.LSTM(self.fea_num, 64, batch_first=True)
        self.rnn_layer2 = nn.LSTM(self.fea_num, 64, batch_first=True)
        self.den = nn.Sequential(
            nn.Linear(64 * num_slot, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
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

    def forward(self, co):
        # co (80, 9, 128, 128)
        co = co.view(-1, 1, self.slot_size, self.slot_size)  # co (80*9, 1, 128, 128)
        co = self.stn_layer(co)
        # co = co.view(-1, 1, self.slot_size, self.slot_size) #co (80*9, 1, 128, 128)
        x1 = self.conv1(co)  # (80*9,3,64,64)
        x2 = self.conv2(x1)  # (80*9,3,32,32)
        x3 = self.conv3(x2)  # (80*9, 3, 16, 16)
        fea = x3.view(-1, 1, self.fea_num)  # (80*9, 1, 3*16*16)

        out = fea.view(-1, self.num_slot, self.fea_num)  # (80, 9, 3*16*16)
        out1, _ = self.rnn_layer1(out)  #
        out2, _ = self.rnn_layer2(out)  # (N, L, 64)
        out2 = torch.flip(out2, [1])  # reverse
        out = torch.add(out1, out2)
        out = torch.flatten(out, start_dim=1)
        out = self.den(out)  # self.den.forward(out)

        return out
