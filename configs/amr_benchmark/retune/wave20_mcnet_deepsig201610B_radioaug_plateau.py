"""Wave-20 Tier-B: RadioAugment (phase+shift) + Keras plateau recipe — mcnet@deepsig201610B (59.49/62)."""
_base_ = ['../../mcnet/mcnet_iq-deepsig-201610B.py']
train_pipeline = [
    dict(type='RadioAugment', key='iq', phase=True, time_shift=8, prob=0.9),
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
                     min_delta=0, patience=50, rule='greater')]
