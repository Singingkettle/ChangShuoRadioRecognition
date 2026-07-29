# Wave 2 P0 retry — early-stop + cosine, inherits 5-ep empirical anchors (bw×20).
_base_ = '../../experiments/jdm-det_fft-csrd_anchor096146_bw20_5ep.py'

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=5e-4),
    clip_grad=dict(max_norm=35, norm_type=2),
)

param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=15,
    eta_min=1e-6,
)

train_cfg = dict(by_epoch=True, max_epochs=15, val_interval=1)

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='detection/mAP',
         min_delta=0.001, patience=5, rule='greater'),
]

work_dir = 'work_dirs/jdm/retune/det_wave2_es_pat5_lr5e4_bw20_tm15'
