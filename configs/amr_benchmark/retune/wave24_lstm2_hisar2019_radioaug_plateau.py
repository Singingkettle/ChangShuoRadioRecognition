"""Wave-24: LSTM2-10A-winning recipe on Hisar — lstm2@hisar2019 (66.81/73)."""
_base_ = ['../../lstm2/lstm2_ap-shape-L-F-hisar-2019.py']
train_pipeline = [
    dict(type='RadioAugment', key='iq', phase=True, time_shift=32, prob=0.9),
    dict(type='IQToAP'),
    dict(type='Transpose', orders=dict(ap=[1, 0])),
    dict(type='PackInputs', input_key='ap'),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
optim_wrapper = dict(optimizer=dict(type='Adam', lr=1e-3))
param_scheduler = dict(
    _delete_=True, type='ReduceOnPlateauParamScheduler', param_name='lr',
    monitor='accuracy/top1', factor=0.5, patience=6, verbose=True,
    rule='greater', min_value=1e-6)
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=1)
custom_hooks = [dict(type='EarlyStoppingHook', monitor='accuracy/top1',
                     min_delta=0, patience=40, rule='greater')]
