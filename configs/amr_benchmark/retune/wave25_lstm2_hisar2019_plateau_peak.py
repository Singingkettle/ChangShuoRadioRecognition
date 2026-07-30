"""Wave-25: lstm2@hisar2019 (66.81/73) — augment-free plateau push.

RadioAugment collapsed lstm2-Hisar twice (raw AP w24: 9.6%, selfnorm+lr5e-4
w25: 8.8%), so augmentation is abandoned for this pair. This run keeps the
baseline pipeline untouched and only adds the plateau LR schedule + patient
early stopping that lifted other Hisar pairs.
"""
_base_ = ['../../lstm2/lstm2_ap-shape-L-F-hisar-2019.py']
optim_wrapper = dict(optimizer=dict(type='Adam', lr=1e-3))
param_scheduler = dict(
    _delete_=True, type='ReduceOnPlateauParamScheduler', param_name='lr',
    monitor='accuracy/top1', factor=0.5, patience=6, verbose=True,
    rule='greater', min_value=1e-6)
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=1)
custom_hooks = [dict(type='EarlyStoppingHook', monitor='accuracy/top1',
                     min_delta=0, patience=40, rule='greater')]
