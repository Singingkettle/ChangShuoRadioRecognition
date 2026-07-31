"""Wave-26: gru2@hisar2019 (68.51/73) — augment-free plateau push.

w24 selfnorm+augment REGRESSED to 61.16 (baseline 68.51). lstm2's augment-free
plateau recipe gained +2.88 on the same dataset; apply it verbatim.
"""
_base_ = ['../../gru2/gru2_iq-shape-L-F-hisar-2019.py']
optim_wrapper = dict(optimizer=dict(type='Adam', lr=1e-3))
param_scheduler = dict(
    _delete_=True, type='ReduceOnPlateauParamScheduler', param_name='lr',
    monitor='accuracy/top1', factor=0.5, patience=6, verbose=True,
    rule='greater', min_value=1e-6)
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=1)
custom_hooks = [dict(type='EarlyStoppingHook', monitor='accuracy/top1',
                     min_delta=0, patience=40, rule='greater')]
