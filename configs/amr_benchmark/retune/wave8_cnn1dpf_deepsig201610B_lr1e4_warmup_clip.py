"""Wave-8 Tier-B: CNN1DPF @ 10B — lower LR 1e-4 + warmup + clip (gap~2.08pp)."""

_base_ = ['./wave4_cnn1dpf_deepsig201610B_lr2e4_warmup.py']

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-4),
    clip_grad=dict(max_norm=5.0, norm_type=2),
)
