import torch
import torch.nn as nn

from .base_head import BaseHead
from ..builder import HEADS, build_head, build_loss

@HEADS.register_module()
class SEDNNHead(BaseHead):
    def __init__(self, head, num_snr, snr_head):
        super(SEDNNHead, self).__init__()

