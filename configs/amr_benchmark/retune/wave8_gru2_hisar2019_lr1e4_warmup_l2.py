"""Wave-8 Tier-B: GRU2 @ Hisar — lr1e-4 + L2 (gap~4.5pp after wave5)."""

_base_ = ['./wave5_gru2_hisar2019_lr2e4_warmup.py']

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=5.0, norm_type=2),
)
