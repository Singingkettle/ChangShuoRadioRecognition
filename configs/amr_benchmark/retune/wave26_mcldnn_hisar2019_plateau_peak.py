"""Wave-26: mcldnn@hisar2019 (69.17/75) — augment-free plateau push."""
_base_ = ['../../mcldnn/mcldnn_iq-hisar-2019.py']
optim_wrapper = dict(optimizer=dict(type='Adam', lr=1e-3))
param_scheduler = dict(
    _delete_=True, type='ReduceOnPlateauParamScheduler', param_name='lr',
    monitor='accuracy/top1', factor=0.5, patience=6, verbose=True,
    rule='greater', min_value=1e-6)
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=1)
custom_hooks = [dict(type='EarlyStoppingHook', monitor='accuracy/top1',
                     min_delta=0, patience=40, rule='greater')]
