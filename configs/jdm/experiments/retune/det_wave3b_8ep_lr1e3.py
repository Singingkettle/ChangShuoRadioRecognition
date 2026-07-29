# Wave 3 Track B — fresh 8-ep train, lr 1e-3, ES off.
_base_ = '../jdm-det_fft-csrd_anchor096146_bw20_5ep.py'

param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=8,
    eta_min=1e-6,
)

train_cfg = dict(by_epoch=True, max_epochs=8, val_interval=1)
custom_hooks = []

work_dir = 'work_dirs/jdm/retune/det_wave3b_8ep_lr1e3'
