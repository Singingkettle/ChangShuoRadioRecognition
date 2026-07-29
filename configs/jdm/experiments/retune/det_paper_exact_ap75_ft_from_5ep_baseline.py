# Paper-exact: AP75 recovery from production 5-ep baseline (AP75 0.9182).
# Architecture freeze — lr/schedule/ES only. Goal: AP75≥0.96 while lifting mAP.
_base_ = '../../experiments/jdm-det_fft-csrd_anchor096146_bw20_5ep.py'

load_from = (
    'work_dirs/jdm/exp_anchor096146_bw20_5ep/best_detection_mAP_epoch_2.pth')

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=2e-5),
    clip_grad=dict(max_norm=35, norm_type=2),
)

param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=10,
    eta_min=1e-7,
)

train_cfg = dict(by_epoch=True, max_epochs=10, val_interval=1)

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='detection/AP75',
         min_delta=0.0005, patience=5, rule='greater'),
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=2,
        save_best=['detection/mAP', 'detection/AP75'],
        rule='greater',
    ),
)

work_dir = 'work_dirs/jdm/retune/det_paper_exact_ap75_ft_from_5ep_baseline'
