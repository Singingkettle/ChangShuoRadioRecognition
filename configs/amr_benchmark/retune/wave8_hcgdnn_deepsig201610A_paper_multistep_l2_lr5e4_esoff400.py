"""Wave-8 Tier-A: HCGDNN — W7 L2 stack @ lr=5e-4 (between 3e-4 and 4.4e-4)."""

_base_ = ['./wave7_hcgdnn_deepsig201610A_paper_multistep_l2_esoff400.py']

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=5e-4, weight_decay=1e-4),
)
