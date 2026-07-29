"""Wave-8 Tier-A: FastMLDNN — softer multi-loss beta=0.3 (was 0.5)."""

_base_ = ['./wave6_fastmldnn_deepsig201610A_paper_fixedlr_l2_esoff500.py']

model = dict(head=dict(beta=0.3))
