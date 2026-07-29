# Wave 3 Track A — fine-tune from 5-ep best, 8 ep, lr 1e-4, ES patience 3.
_base_ = '../../experiments/jdm-det_fft-csrd_anchor096146_bw20_5ep.py'

load_from = (
    'work_dirs/jdm/exp_anchor096146_bw20_5ep/best_detection_mAP_epoch_2.pth')

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-4),
    clip_grad=dict(max_norm=35, norm_type=2),
)

param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=8,
    eta_min=1e-6,
)

train_cfg = dict(by_epoch=True, max_epochs=8, val_interval=1)

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='detection/mAP',
         min_delta=0.001, patience=3, rule='greater'),
]

work_dir = 'work_dirs/jdm/retune/det_wave3_ft_8ep_lr1e4_es3'
