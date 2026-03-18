from .cls_head import ClsHead
from .dae_head import DAEHead
from .fast_mldnn_head import FastMLDNNHead
from .hcgdnn_head import HCGDNNHead
from .mldnn_head import MLDNNHead
from .multi_task_head import MultiTaskHead
from .snr_auxiliary_head import SNRAuxiliaryHead

__all__ = [
    'ClsHead', 'DAEHead', 'FastMLDNNHead',
    'HCGDNNHead', 'MLDNNHead', 'MultiTaskHead',
    'SNRAuxiliaryHead',
]
