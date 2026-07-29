# P1 AMC push toward 90%: 60-ep FT from wave3b 30ep best (test ~83%).
# Skip failed 5ep AP75 FT / 5ep+AMC AWGN merge paths.
_base_ = './amc_wave3b_detprops_30ep.py'

load_from = (
    'work_dirs/jdm/retune/amc_wave3b_detprops_30ep/'
    'best_accuracy_top1_epoch_23.pth')

param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=60,
    eta_min=1e-6,
)

optim_wrapper = dict(optimizer=dict(type='Adam', lr=5e-4))

train_cfg = dict(by_epoch=True, max_epochs=60, val_interval=1)
work_dir = 'work_dirs/jdm/retune/amc_wave3b_detprops_60ep_ft'
