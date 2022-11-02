import logging

import torch
import torch.nn as nn

from .crnet import CRNet
from ..builder import BACKBONES
from ...runner import load_checkpoint


@BACKBONES.register_module()
class DSCLNet(nn.Module):

    def __init__(self, in_channels, cnn_depth, rnn_depth, input_size, in_height=2, avg_pool=None, out_indices=(1,),
                 is_last=True, rnn_mode='LSTM', fusion_mode='bmm'):
        super(DSCLNet, self).__init__()
        self.fusion = fusion_mode + '_fusion'
        self.iq_crnet = CRNet(in_channels, cnn_depth, rnn_depth, input_size, in_height=in_height,
                              avg_pool=avg_pool, out_indices=out_indices, is_last=is_last, rnn_mode=rnn_mode)
        self.ap_crnet = CRNet(in_channels, cnn_depth, rnn_depth, input_size, in_height=in_height,
                              avg_pool=avg_pool, out_indices=out_indices, is_last=is_last, rnn_mode=rnn_mode)

    def init_weights(self, pre_trained=None):
        if isinstance(pre_trained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pre_trained, strict=False, logger=logger)
        elif pre_trained is None:
            self.iq_crnet.init_weights()
            self.ap_crnet.init_weights()

    def bmm_fusion(self, iq, ap):
        iq = torch.unsqueeze(iq, 2)
        ap = torch.unsqueeze(ap, 1)  # (N, 1, C)

        # (N, C, C) Please refer the https://pytorch.org/docs/0.2.0/torch.html#torch.bmm for the detail of torch.bmm
        x = torch.bmm(iq, ap)
        x = torch.reshape(x, (-1, 2500, 1, 1))

        return x

    def cat_fusion(self, iq, ap):

        return torch.cat((iq, ap), dim=1)

    def add_fusion(self, iq, ap):

        return torch.add(iq, ap)

    def sub_fusion(self, iq, ap):

        return torch.sub(iq, ap)

    def mul_fusion(self, iq, ap):

        return torch.mul(iq, ap)

    def forward(self, iqs, aps):
        iq = self.iq_crnet(iqs)
        ap = self.ap_crnet(aps)
        fusion = getattr(self, self.fusion)
        x = fusion(iq, ap)

        return x
