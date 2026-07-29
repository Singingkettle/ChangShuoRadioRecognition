"""Wave-7 Tier-A: FastMLDNN @ 10A — W6 stack with lighter weight_decay=5e-5."""

_base_ = ['./wave6_fastmldnn_deepsig201610A_paper_fixedlr_l2_esoff500.py']

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.00044, weight_decay=5e-5),
)
