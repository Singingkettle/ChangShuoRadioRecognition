# Wave-21 detector: EMA-stabilized training to beat the unstable early peak.
#
# det_full_90ep / 120ep both peak at epoch 3-4 then degrade -> the raw-weight
# val mAP is noisy. Train a fresh detector (same paper recipe) with an EMA of
# the weights so the evaluated model is the smoothed trajectory, which usually
# lifts and stabilizes the peak. Architecture / losses unchanged.
_base_ = '../../jdm-det_fft-csrd.py'

param_scheduler = dict(
    type='CosineAnnealingLR', by_epoch=True, T_max=120, eta_min=1e-6)
train_cfg = dict(by_epoch=True, max_epochs=120, val_interval=1)

custom_hooks = [
    dict(type='EMAHook', ema_type='ExponentialMovingAverage',
         momentum=0.0004, update_buffers=True, priority=49),
]

work_dir = 'work_dirs/jdm/retune/det_ema_stable_w21'
