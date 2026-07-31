"""Wave-28: cnn1dpf@deepsig201610A (54.97/pass 55.5) — es_patience30 recipe.

The augment attempt regressed (53.88); its 10B sibling's best came from the
plain patient-early-stop recipe instead, so port that here.
"""
_base_ = ['../../cnn1dpf/cnn1dpf_iq-deepsig-201610A.py']
optim_wrapper = dict(optimizer=dict(type='Adam', lr=1e-3))
param_scheduler = dict(
    _delete_=True, type='ReduceOnPlateauParamScheduler', param_name='lr',
    monitor='accuracy/top1', factor=0.5, patience=8, verbose=True,
    rule='greater', min_value=1e-6)
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=1)
custom_hooks = [dict(type='EarlyStoppingHook', monitor='accuracy/top1',
                     min_delta=0, patience=30, rule='greater')]
