# Operating-point detector: same architecture as jdm-det_fft-csrd.py, 120-epoch
# cosine. Best checkpoint is early (epoch 4). Evaluate with the protocol configs
# in this directory, not the mixed 124-version test split.
# Paper: Xing et al., "Joint Signal Detection and Automatic Modulation
# Classification via Deep Learning", IEEE TWC 2024.
_base_ = '../jdm-det_fft-csrd.py'

param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=120,
    eta_min=1e-6,
)

train_cfg = dict(by_epoch=True, max_epochs=120, val_interval=1)

work_dir = 'work_dirs/jdm/retune/det_full_120ep_lr1e3'
