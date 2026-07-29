"""Wave-8 Tier-A: HCGDNN — CosineAnnealing + L2 (contrast MultiStep)."""

_base_ = ['./wave7_hcgdnn_deepsig201610A_paper_multistep_l2_esoff400.py']

param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=True, begin=0, end=10,
         convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', by_epoch=True, T_max=390, begin=10, end=400,
         eta_min=1e-6),
]
