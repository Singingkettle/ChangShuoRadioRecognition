"""Wave-6 Tier-B: ResNetAMR @ 10B — moderate LR (gap 0.13pp overall after wave4)."""

_base_ = ['../../resnetamr/resnetamr_iq-deepsig-201610B.py']

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=5e-4),
    clip_grad=dict(max_norm=5.0, norm_type=2),
)

param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=150,
    eta_min=1e-6,
)
