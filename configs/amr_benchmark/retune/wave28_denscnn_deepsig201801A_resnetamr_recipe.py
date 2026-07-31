"""Wave-28: denscnn@deepsig201801A (53.99/pass 56.5) — resnetamr-2018 winning recipe.

resnetamr-201801A passed (57.10, +1.86 over its baseline) with RadioAugment
shift32 p=0.9 + plateau; port the exact recipe to the sibling CNN.
"""
_base_ = ['../../denscnn/denscnn_iq-deepsig-201801A.py']
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
