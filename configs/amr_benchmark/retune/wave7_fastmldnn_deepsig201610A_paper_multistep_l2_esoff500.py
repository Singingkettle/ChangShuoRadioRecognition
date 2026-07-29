"""Wave-7 Tier-A: FastMLDNN @ 10A — paper MultiStep + L2, 500ep (W6 was fixed-LR)."""

_base_ = ['./wave4_fastmldnn_deepsig201610A_paper_multistep_beta05_dp007_esoff400.py']

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.00044, weight_decay=1e-4),
)

train_cfg = dict(by_epoch=True, max_epochs=500, val_interval=1)

param_scheduler = dict(
    _delete_=True,
    type='MultiStepLR',
    by_epoch=True,
    milestones=[25, 100, 250, 375, 475],
    gamma=0.3,
)
