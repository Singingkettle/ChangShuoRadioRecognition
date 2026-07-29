"""Wave-8 Tier-A: FastMLDNN — combine W7 best pieces (wd5e5 + 600ep).

W7 best near-miss: paper_fixedlr_l2_esoff600 60.89/92.14 (−0.13pp peak+);
wd5e5 alone 60.74. Combine both; architecture freeze.
"""

_base_ = ['./wave6_fastmldnn_deepsig201610A_paper_fixedlr_l2_esoff500.py']

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.00044, weight_decay=5e-5),
)

train_cfg = dict(by_epoch=True, max_epochs=600, val_interval=1)
