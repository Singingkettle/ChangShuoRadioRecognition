# Paper-exact campaign: AP75 recovery FT from Wave3B best (mAP 0.8113).
# Architecture freeze — only lr/schedule/epochs/ES.
_base_ = '../../experiments/jdm-det_fft-csrd_anchor096146_bw20_5ep.py'

load_from = (
    'work_dirs/jdm/retune/det_wave3b_5ep_lr1e3/best_detection_mAP_epoch_5.pth')

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=5e-5),
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
    dict(type='EarlyStoppingHook', monitor='detection/AP75',
         min_delta=0.001, patience=4, rule='greater'),
]

# Prefer checkpoints that raise AP75 while keeping mAP; save_best stays mAP
# from base CheckpointHook; we also monitor AP75 via ES.
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=2,
        save_best=['detection/mAP', 'detection/AP75'],
        rule='greater',
    ),
)

work_dir = 'work_dirs/jdm/retune/det_paper_exact_ap75_ft_from_wave3b'
