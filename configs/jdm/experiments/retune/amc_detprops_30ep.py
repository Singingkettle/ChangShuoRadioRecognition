# 30-epoch proposal-crop AMC fine-tune from 20-ep best.
_base_ = '../../experiments/jdm-amc_iq-csrd_detprops_20ep.py'

load_from = 'work_dirs/jdm/exp_amc_detprops_20ep/best_accuracy_top1_epoch_20.pth'

param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=30,
    eta_min=1e-6,
)

train_cfg = dict(by_epoch=True, max_epochs=30, val_interval=1)
work_dir = 'work_dirs/jdm/retune/amc_detprops_30ep'
