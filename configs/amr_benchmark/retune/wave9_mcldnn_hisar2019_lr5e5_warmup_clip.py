"""Wave-9 Tier-B: MCLDNN @ Hisar — lr5e-5 + warmup + clip (W8 used 1e-4+l2)."""

_base_ = ['./wave8_mcldnn_hisar2019_lr1e4_warmup_l2.py']

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=5e-5, weight_decay=1e-4),
    clip_grad=dict(max_norm=5.0, norm_type=2),
)
