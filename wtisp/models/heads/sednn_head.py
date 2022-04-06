from .base_head import BaseHead
from ..builder import HEADS


@HEADS.register_module()
class SEDNNHead(BaseHead):
    def __init__(self, head, num_snr, snr_head):
        super(SEDNNHead, self).__init__()
        self.num_str = num_snr
        self.a = a
