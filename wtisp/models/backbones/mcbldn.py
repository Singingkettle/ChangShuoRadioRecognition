import logging
import torch
import torch.nn as nn

from ..builder import BACKBONES
from ...runner import load_checkpoint


@BACKBONES.register_module()
class ScdnNet(nn.Module):

    def __init__(self, slot_size=128):
        super(ScdnNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv3 = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(16, 32, kernel_size=(2, 2), stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv4 = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(32, 16, kernel_size=(2, 2), stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2))
        self.den1 = nn.Sequential(
            nn.Linear(int(slot_size * slot_size / 4), 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2))
        self.den2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2))
        self.den3 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2))

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
        x1 = self.conv1(co)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = torch.flatten(x4, start_dim=1)
        x6 = self.den1(x5)
        x7 = self.den2(x6)
        x8 = self.den3(x7)
        return x8


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


@BACKBONES.register_module()
class McbldnNet(nn.Module):
    def __init__(self, num_slot, num_filter, slot_size):
        super(McbldnNet, self).__init__()
        self.slotlayers = []
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
