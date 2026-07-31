"""Wave-29: cnn4@deepsig201801A — port resnetamr-2018 winning RadioAugment recipe.

Baseline 54.55 overall / 84.57 peak (pass 53.5/90.0): overall OK, peak short.
"""
_base_ = ['../../cnn4/cnn4_iq-deepsig-201801A.py']
train_pipeline = [
    dict(type='RadioAugment', key='iq', phase=True, time_shift=32, prob=0.9),
    dict(type='Reshape', shapes=dict(iq=[1, 2, 1024])),
    dict(type='PackInputs', input_key='iq'),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
optim_wrapper = dict(optimizer=dict(type='Adam', lr=1e-3))
param_scheduler = dict(
    _delete_=True, type='ReduceOnPlateauParamScheduler', param_name='lr',
    monitor='accuracy/top1', factor=0.5, patience=5, verbose=True,
    rule='greater', min_value=1e-6)
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
custom_hooks = [dict(type='EarlyStoppingHook', monitor='accuracy/top1',
                     min_delta=0, patience=20, rule='greater')]
