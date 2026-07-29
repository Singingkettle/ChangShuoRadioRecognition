# Ideal-protocol detector train: CSRD v1, longer schedule (architecture freeze).
# Prior ideal 30ep@1e-3 best test mAP 0.385@ep7; try gentler LR + 60ep.
_base_ = './det_ideal_v1_30ep.py'

param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=60,
    eta_min=1e-6,
)

optim_wrapper = dict(optimizer=dict(type='Adam', lr=5e-4))

train_cfg = dict(by_epoch=True, max_epochs=60, val_interval=1)

work_dir = 'work_dirs/jdm/retune/det_ideal_v1_60ep_lr5e4'
