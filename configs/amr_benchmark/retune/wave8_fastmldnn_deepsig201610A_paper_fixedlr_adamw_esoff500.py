"""Wave-8 Tier-A: FastMLDNN — AdamW instead of Adam (L2 via decoupled WD)."""

_base_ = ['./wave6_fastmldnn_deepsig201610A_paper_fixedlr_l2_esoff500.py']

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.00044, weight_decay=1e-4),
)
