"""Wave-8 Tier-A: HCGDNN — W7 best MultiStep+L2 with lighter wd=5e-5."""

_base_ = ['./wave7_hcgdnn_deepsig201610A_paper_multistep_l2_esoff400.py']

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.00044, weight_decay=5e-5),
)
