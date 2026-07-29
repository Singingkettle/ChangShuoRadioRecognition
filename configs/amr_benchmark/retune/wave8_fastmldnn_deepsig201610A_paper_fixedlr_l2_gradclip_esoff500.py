"""Wave-8 Tier-A: FastMLDNN — W6 stack + grad clip max_norm=5."""

_base_ = ['./wave6_fastmldnn_deepsig201610A_paper_fixedlr_l2_esoff500.py']

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.00044, weight_decay=1e-4),
    clip_grad=dict(max_norm=5.0, norm_type=2),
)
