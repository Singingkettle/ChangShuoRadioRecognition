# Copyright (c) OpenMMLab. All rights reserved.
from .beitv1_head import BEiTV1Head
from .beitv2_head import BEiTV2Head
from .cae_head import CAEHead
from .cls_head import ClsHead
from .conformer_head import ConformerHead
from .contrastive_head import ContrastiveHead
from .dae_head import DAEHead
from .deit_head import DeiTClsHead
from .efficientformer_head import EfficientFormerClsHead
from .fast_mldnn_head import FastMLDNNHead
from .grounding_head import GroundingHead
from .hcgdnn_head import HCGDNNHead
from .itc_head import ITCHead
from .itm_head import ITMHead
from .latent_heads import LatentCrossCorrelationHead, LatentPredictHead
from .levit_head import LeViTClsHead
from .linear_head import LinearClsHead
from .mae_head import MAEPretrainHead
from .margin_head import ArcFaceClsHead
from .mim_head import MIMHead
from .mixmim_head import MixMIMPretrainHead
from .mldnn_head import MLDNNHead
from .mocov3_head import MoCoV3Head
from .multi_label_cls_head import MultiLabelClsHead
from .multi_label_csra_head import CSRAClsHead
from .multi_label_linear_head import MultiLabelLinearClsHead
from .multi_task_head import MultiTaskHead
from .seq_gen_head import SeqGenerationHead
from .simmim_head import SimMIMHead
from .snr_auxiliary_head import SNRAuxiliaryHead
from .stacked_head import StackedLinearClsHead
from .swav_head import SwAVHead
from .vig_head import VigClsHead
from .vision_transformer_head import VisionTransformerClsHead
from .vqa_head import VQAGenerationHead

__all__ = [
    'DAEHead',
    'MLDNNHead',
    'ClsHead',
    'LinearClsHead',
    'StackedLinearClsHead',
    'MultiLabelClsHead',
    'MultiLabelLinearClsHead',
    'VisionTransformerClsHead',
    'DeiTClsHead',
    'ConformerHead',
    'EfficientFormerClsHead',
    'ArcFaceClsHead',
    'CSRAClsHead',
    'MultiTaskHead',
    'LeViTClsHead',
    'VigClsHead',
    'BEiTV1Head',
    'BEiTV2Head',
    'CAEHead',
    'ContrastiveHead',
    'LatentCrossCorrelationHead',
    'LatentPredictHead',
    'MAEPretrainHead',
    'MixMIMPretrainHead',
    'SwAVHead',
    'MoCoV3Head',
    'MIMHead',
    'SimMIMHead',
    'SeqGenerationHead',
    'VQAGenerationHead',
    'ITCHead',
    'ITMHead',
    'GroundingHead',
]
