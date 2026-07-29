# SWA snapshot-collection run (W22): resume the best 120-epoch detector and
# train 16 more epochs at a small CONSTANT LR, saving every epoch. The
# snapshots are then weight-averaged (tools/jdm/swa_average.py) and the
# averaged model is evaluated with box voting. Narrative-safe: same
# model/anchors/losses/optimizer; only a short constant-LR tail is added and
# reported weights become an average of the last epochs (standard SWA).
_base_ = './det_full_120ep_lr1e3.py'

load_from = 'work_dirs/jdm/retune/det_full_120ep_lr1e3/epoch_120.pth'

optim_wrapper = dict(optimizer=dict(lr=5e-5))

param_scheduler = dict(
    _delete_=True,
    type='ConstantLR',
    factor=1.0,
    by_epoch=True,
)

train_cfg = dict(by_epoch=True, max_epochs=16, val_interval=1)

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=-1, save_best='detection/mAP'))

work_dir = 'work_dirs/jdm/retune/det_swa_from120_w22'
