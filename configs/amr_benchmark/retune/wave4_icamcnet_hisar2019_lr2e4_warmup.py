"""Wave-4 marginal retune: ICAMCNet @ HisarMod — lower LR + warmup + grad clip."""

_base_ = ['../../icamcnet/icamcnet_iq-hisar-2019.py']

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=2e-4),
    clip_grad=dict(max_norm=5.0, norm_type=2),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=True, begin=0, end=5,
         convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', by_epoch=True, T_max=145, begin=5, end=150,
         eta_min=1e-6),
]
