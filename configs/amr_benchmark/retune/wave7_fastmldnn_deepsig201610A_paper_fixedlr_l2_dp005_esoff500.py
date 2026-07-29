"""Wave-7 Tier-A: FastMLDNN @ 10A — W6 stack with softer dp=0.05."""

_base_ = ['./wave6_fastmldnn_deepsig201610A_paper_fixedlr_l2_esoff500.py']

model = dict(
    backbone=dict(dp=0.05),
)
