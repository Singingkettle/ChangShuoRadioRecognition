# Full-data detector, 90-epoch cosine with a gentler peak LR (5e-4 vs base 1e-3)
# (architecture freeze). A different training RECIPE — longer schedule + lower
# peak learning rate — to probe whether a smoother descent closes the ideal-det
# gap (v1 test-only eval ~0.80 vs paper 0.91). SAME model/anchors/losses and
# optimizer TYPE (Adam) as jdm-det_fft-csrd.py; only the optimizer lr, schedule
# length and max_epochs change. Evaluate on v1 test-only via
# eval_ideal_v1_det_testonly.py.
_base_ = '../../jdm-det_fft-csrd.py'

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=5e-4),
    clip_grad=dict(max_norm=35, norm_type=2),
)

param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=90,
    eta_min=1e-6,
)

train_cfg = dict(by_epoch=True, max_epochs=90, val_interval=1)

work_dir = 'work_dirs/jdm/retune/det_full_90ep_lr5e4'
