"""Wave-25 retry: cnn2@deepsig201610B (56.25/64) after raw-IQ collapse (10%).

Same failure mode as cldnnl/cldnnw: phase augment on raw un-normalized IQ
collapses training. Fix: SelfNormalize after RadioAugment.
"""
_base_ = ['../../cnn2/cnn2_iq-deepsig-201610B.py']
train_pipeline = [
    dict(type='RadioAugment', key='iq', phase=True, time_shift=8, prob=0.9),
    dict(type='SelfNormalize', norms=dict(iq={})),
    dict(type='Reshape', shapes=dict(iq=[1, 2, 128])),
    dict(type='PackInputs', input_key='iq'),
]
val_pipeline = [
    dict(type='SelfNormalize', norms=dict(iq={})),
    dict(type='Reshape', shapes=dict(iq=[1, 2, 128])),
    dict(type='PackInputs', input_key='iq'),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=val_pipeline))
test_dataloader = dict(dataset=dict(pipeline=val_pipeline))
optim_wrapper = dict(optimizer=dict(type='Adam', lr=1e-3))
param_scheduler = dict(
    _delete_=True, type='ReduceOnPlateauParamScheduler', param_name='lr',
    monitor='accuracy/top1', factor=0.5, patience=8, verbose=True,
    rule='greater', min_value=1e-6)
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=1)
custom_hooks = [dict(type='EarlyStoppingHook', monitor='accuracy/top1',
                     min_delta=0, patience=50, rule='greater')]
