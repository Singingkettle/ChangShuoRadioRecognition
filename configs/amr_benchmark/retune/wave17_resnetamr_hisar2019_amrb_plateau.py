"""Wave-17 Tier-B (HisarMod): AMR-Benchmark ORIGINAL Keras recipe.

resnetamr @ hisar2019: measured 72.49/80 gap -7.51. Same recipe that produced the CLDNNL-10B and
LSTM2-10A passes: Adam 1e-3 + ReduceLROnPlateau(f0.5 p5) + ES(p50).
"""
_base_ = ['../../resnetamr/resnetamr_iq-hisar-2019.py']

optim_wrapper = dict(optimizer=dict(type='Adam', lr=1e-3))

param_scheduler = dict(
    _delete_=True,
    type='ReduceOnPlateauParamScheduler',
    param_name='lr',
    monitor='accuracy/top1',
    factor=0.5,
    patience=5,
    verbose=True,
    rule='greater',
    min_value=1e-6,
)

train_cfg = dict(by_epoch=True, max_epochs=400, val_interval=1)

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='accuracy/top1', min_delta=0,
         patience=50, rule='greater'),
]
