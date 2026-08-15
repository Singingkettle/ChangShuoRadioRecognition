import torch
import torch.nn as nn

from .base_backbone import BaseBackbone
from ..builder import BACKBONES



@BACKBONES.register_module()
class CLDNNL(BaseBackbone):
    """`CLDNNL <https://ieeexplore.ieee.org/abstract/document/8335483>`_ backbone
    Actually, the details of neural network structure is not provided in the paper.
    To deal with that, we have referred to the code OF CLDNN2 in the AMR-Benchmark.
    Basically, there are two versions of CLDNN. In order to identify them, we add an
    extra letter after CLDNN, which is borrowed from the first name of the first author.
    The input for CNN1 is a 1*2*L frame
    Args:
        frame_length (int): the frame length equal to number of sample points
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, frame_length=128, num_classes=-1, init_cfg=None):
        # Match the AMR-Benchmark Keras CLDNN2 reference (`glorot_uniform`).
        # The deep 256-channel conv stack with Dropout(0.5) after every layer
        # vanishes under PyTorch's default init and freezes the network at
        # random chance (loss == ln(num_classes)); Xavier-uniform restores a
        # healthy gradient signal.
        if init_cfg is None:
            init_cfg = [
                dict(type='Xavier', layer='Conv2d', distribution='uniform'),
                dict(type='Xavier', layer='Linear', distribution='uniform'),
            ]
        super(CLDNNL, self).__init__(init_cfg=init_cfg)
        self.frame_length = frame_length
        self.num_classes = num_classes
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3), padding='valid'),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=(2, 3), padding='valid'),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 80, kernel_size=(1, 3), padding='valid'),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(80, 80, kernel_size=(1, 3), padding='valid'),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.lstm = nn.LSTM(input_size=self.frame_length - 8, hidden_size=50, batch_first=True)

        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(50, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes),
            )

    def forward(self, x):

        x = self.cnn(x)
        x = torch.reshape(x, [-1, 80, self.frame_length - 8])
        x, _ = self.lstm(x)
        if self.num_classes > 0:
            x = self.classifier(x[:, -1, :])

        return (x,)



@BACKBONES.register_module()
class CLDNNW(BaseBackbone):
    """`CLDNNW <https://ieeexplore.ieee.org/abstract/document/7920754>`_ backbone
    Actually, the details of neural network structure is not provided in the paper.
    To deal with that, we have referred to the code of CLDNN in the AMR-Benchmark.
    Basically, there are two versions of CLDNN. In order to identify them, we add an
    extra letter after CLDNN, which is borrowed from the first name of the first author.
    The input for CNN1 is a 1*2*L frame
    Args:
        frame_length (int): the frame length equal to number of sample points
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, frame_length=128, num_classes=-1, use_zero_pad=True,
                 init_cfg=None):
        super(CLDNNW, self).__init__(init_cfg=init_cfg)
        self.frame_length = frame_length
        self.num_classes = num_classes
        self.use_zero_pad = use_zero_pad
        # TF AMR-Benchmark CLDNN uses ZeroPadding2D((0, 2)) before each of the
        # three (1, 8) convs (channels_first). That shrinks width by 3 per conv
        # and yields LSTM input_size 488 @ L=128. CSRR previously dropped the
        # pads (shrink-by-7, LSTM 456); ``use_zero_pad=True`` restores the TF
        # topology. Set False only to load legacy no-pad checkpoints.
        # nn.ZeroPad2d((left, right, top, bottom)) ≡ TF pad on W by 2 each side.
        if use_zero_pad:
            self.cnn1 = nn.Sequential(
                nn.ZeroPad2d((2, 2, 0, 0)),
                nn.Conv2d(1, 50, kernel_size=(1, 8), padding='valid'),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
            )
            self.cnn2 = nn.Sequential(
                nn.ZeroPad2d((2, 2, 0, 0)),
                nn.Conv2d(50, 50, kernel_size=(1, 8), padding='valid'),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.ZeroPad2d((2, 2, 0, 0)),
                nn.Conv2d(50, 50, kernel_size=(1, 8), padding='valid'),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
            )
            # 2*(L-3) + 2*(L-9) = (2L - 12)*2 = (L*2 - 3*4)*2
            self._lstm_feat = (self.frame_length * 2 - 3 * 4) * 2
        else:
            self.cnn1 = nn.Sequential(
                nn.Conv2d(1, 50, kernel_size=(1, 8), padding='valid'),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
            )
            self.cnn2 = nn.Sequential(
                nn.Conv2d(50, 50, kernel_size=(1, 8), padding='valid'),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Conv2d(50, 50, kernel_size=(1, 8), padding='valid'),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
            )
            self._lstm_feat = (self.frame_length * 2 - 7 * 4) * 2
        self.lstm = nn.LSTM(input_size=self._lstm_feat, hidden_size=50,
                            batch_first=True)

        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(50, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes),
            )

    def forward(self, x):

        x1 = self.cnn1(x)
        x2 = self.cnn2(x1)
        x = torch.concatenate((x1, x2), dim=3)
        x = torch.reshape(x, [-1, 50, self._lstm_feat])
        x, _ = self.lstm(x)
        if self.num_classes > 0:
            x = self.classifier(x[:, -1, :])

        return (x,)



