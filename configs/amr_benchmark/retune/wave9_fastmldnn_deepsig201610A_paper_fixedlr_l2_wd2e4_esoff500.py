"""Wave-9 Tier-A: FastMLDNN — stronger WD=2e-4 (tried 1e-4 and 5e-5)."""

_base_ = ['./wave6_fastmldnn_deepsig201610A_paper_fixedlr_l2_esoff500.py']

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.00044, weight_decay=2e-4),
)

train_cfg = dict(by_epoch=True, max_epochs=500, val_interval=1)
