"""Wave-82 auto Tier-B: ResNetAMR @ 10B."""
_base_ = ['./wave6_resnetamr_deepsig201610B_lr5e4.py']
optim_wrapper = dict(optimizer=dict(type='Adam', lr=1e-4),
                     clip_grad=dict(max_norm=5.0, norm_type=2))
