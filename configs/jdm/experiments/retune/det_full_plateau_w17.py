# Full-data detector with the ADAPTIVE PLATEAU recipe (architecture freeze;
# only schedule/stopping change vs the cosine rungs, which saturated ideal-det
# v1 test-only at 0.8027 vs paper 0.91). ReduceOnPlateau on detection/mAP +
# patient ES — the recipe behind the AMC/Tier-A/Tier-B breakthroughs.
_base_ = '../../jdm-det_fft-csrd.py'

param_scheduler = dict(
    _delete_=True,
    type='ReduceOnPlateauParamScheduler',
    param_name='lr',
    monitor='detection/mAP',
    factor=0.5,
    patience=4,
    verbose=True,
    rule='greater',
    min_value=1e-7,
)

train_cfg = dict(by_epoch=True, max_epochs=150, val_interval=1)

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='detection/mAP', min_delta=0,
         patience=15, rule='greater'),
]

work_dir = 'work_dirs/jdm/retune/det_full_plateau_w17'
