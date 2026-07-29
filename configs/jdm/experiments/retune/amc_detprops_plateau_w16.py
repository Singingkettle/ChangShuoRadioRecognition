# AMC proposal-crop fine-tune with the ADAPTIVE PLATEAU recipe (architecture
# freeze; only schedule/stopping change vs amc_wave3b_detprops_30ep).
# Motivation: ReduceOnPlateau + patient ES is the recipe that produced the
# HCGDNN/FastMLDNN Tier-A breakthroughs and the CLDNNL-10B Tier-B pass; the
# fixed cosine rungs plateaued AMC at 83.27 (target 90).
# PREREQUISITES: same as the other amc rungs (proposal cache + 20ep ckpt).
_base_ = 'amc_wave3b_detprops_30ep.py'

param_scheduler = dict(
    _delete_=True,
    type='ReduceOnPlateauParamScheduler',
    param_name='lr',
    monitor='accuracy/top1',
    factor=0.5,
    patience=8,
    verbose=True,
    rule='greater',
    min_value=1e-7,
)

train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=1)

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='accuracy/top1', min_delta=0,
         patience=40, rule='greater'),
]

work_dir = 'work_dirs/jdm/retune/amc_detprops_plateau_w16'
