"""Wave-7 Tier-A: FastMLDNN @ 10A — extend W6 near-miss (−0.45pp) to 600ep."""

_base_ = ['./wave6_fastmldnn_deepsig201610A_paper_fixedlr_l2_esoff500.py']

train_cfg = dict(by_epoch=True, max_epochs=600, val_interval=1)
