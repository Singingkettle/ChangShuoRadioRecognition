"""Wave-9 Tier-A: FastMLDNN — stronger multi-loss beta=0.7 (W8 tried softer 0.3)."""

_base_ = ['./wave6_fastmldnn_deepsig201610A_paper_fixedlr_l2_esoff500.py']

model = dict(head=dict(beta=0.7))

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.00044, weight_decay=1e-4),
)

train_cfg = dict(by_epoch=True, max_epochs=500, val_interval=1)
