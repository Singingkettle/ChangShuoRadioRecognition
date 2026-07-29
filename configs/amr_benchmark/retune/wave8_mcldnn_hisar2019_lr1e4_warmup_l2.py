"""Wave-8 Tier-B: MCLDNN @ Hisar — lr1e-4 + L2 (gap~2.84pp)."""

_base_ = ['./wave4_mcldnn_hisar2019_lr2e4_warmup.py']

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=5.0, norm_type=2),
)
