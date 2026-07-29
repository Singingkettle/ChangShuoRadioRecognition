"""Wave-1 retune: CNN1DPF @ RML2018.01A — lower LR + warmup + grad clip."""

_base_ = ['../../cnn1dpf/cnn1dpf_iq-deepsig-201801A.py']

model = dict(
    backbone=dict(
        init_cfg=[
            dict(type='Xavier', layer='Conv1d', distribution='uniform'),
            dict(type='Xavier', layer='Linear', distribution='uniform'),
        ],
    ),
)

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
