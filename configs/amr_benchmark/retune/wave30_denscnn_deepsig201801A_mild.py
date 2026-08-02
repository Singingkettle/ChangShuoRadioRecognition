"""Wave-30: denscnn@201801A mild phase (resnetamr-10B winning mild recipe).

Current best 54.19/90.46 (pass 56.5/91.0): peak nearly OK, overall short.
"""
_base_ = ['../../denscnn/denscnn_iq-deepsig-201801A.py']
train_pipeline = [
    dict(type='RadioAugment', key='iq', phase=True, time_shift=0, prob=0.5),
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
