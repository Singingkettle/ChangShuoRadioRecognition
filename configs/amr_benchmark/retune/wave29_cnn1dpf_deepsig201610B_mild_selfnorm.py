"""Wave-29: cnn1dpf@deepsig201610B — mild phase + SelfNormalize on best ES recipe.

Baseline best 58.88/89.41 (pass 60.5/87). Peak already OK; need overall.
"""
_base_ = ['../../cnn1dpf/cnn1dpf_iq-deepsig-201610B.py']
train_pipeline = [
    dict(type='RadioAugment', key='iq', phase=True, time_shift=0, prob=0.5),
    dict(type='SelfNormalize', norms=dict(iq={})),
    dict(type='Reshape', shapes=dict(iq=[1, 2, 128])),
    dict(type='PackInputs', input_key='iq'),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
optim_wrapper = dict(optimizer=dict(type='Adam', lr=1e-3))
param_scheduler = dict(
    _delete_=True, type='ReduceOnPlateauParamScheduler', param_name='lr',
    monitor='accuracy/top1', factor=0.5, patience=8, verbose=True,
    rule='greater', min_value=1e-6)
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=1)
custom_hooks = [dict(type='EarlyStoppingHook', monitor='accuracy/top1',
                     min_delta=0, patience=30, rule='greater')]
