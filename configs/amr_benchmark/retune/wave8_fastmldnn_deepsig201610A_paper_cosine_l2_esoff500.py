"""Wave-8 Tier-A: FastMLDNN — CosineAnnealing vs W6/W7 ConstantLR."""

_base_ = ['./wave6_fastmldnn_deepsig201610A_paper_fixedlr_l2_esoff500.py']

param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=True, begin=0, end=10,
         convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', by_epoch=True, T_max=490, begin=10, end=500,
         eta_min=1e-6),
]
