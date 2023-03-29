from .amc_head import AMCHead
from .base_head import BaseHead
from .center_head import CenterHead
from .classification_head import ClassificationHead
from .cosine_head import CosineHead
from .dscldnn_head import DSCLDNNHead
from .euclidean_head import EuclideanHead
from .fast_mldnn_head import FastMLDNNHead
from .gb_head import GBBCEHead, GBIndHead, GBDetHead
from .hcgdnn_head import HCGDNNHead
from .mldnn_head import MergeAMCHead, MLDNNHead
from .mm_head import ASDHead, ASSHead
from .sednn_head import SEDNNHead
from .shrinkage_head import ShrinkageHead
from .ss_head import SSHead

__all__ = [
    'BaseHead',
    'AMCHead',
    'CenterHead',
    'ClassificationHead',
    'CosineHead',
    'DSCLDNNHead',
    'FastMLDNNHead',
    'GBBCEHead', 'GBIndHead', 'GBDetHead',
    'HCGDNNHead',
    'MergeAMCHead',
    'MLDNNHead',
    'SEDNNHead',
    'ShrinkageHead',
    'ASDHead',
    'ASSHead',
    'SSHead',
    'EuclideanHead'
]
