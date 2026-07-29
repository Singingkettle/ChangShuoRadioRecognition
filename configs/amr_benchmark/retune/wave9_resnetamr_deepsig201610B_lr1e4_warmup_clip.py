"""Wave-9 Tier-B: ResNetAMR @ 10B — lr1e-4 + warmup + clip (gap~ after W6 es_patience40)."""

_base_ = ['./wave6_resnetamr_deepsig201610B_lr5e4.py']

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-4),
    clip_grad=dict(max_norm=5.0, norm_type=2),
)
