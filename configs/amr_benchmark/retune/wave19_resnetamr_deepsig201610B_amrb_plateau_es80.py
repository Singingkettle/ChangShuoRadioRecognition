"""Wave-19 Tier-B: Keras recipe with longer ES (p80) — resnetamr@deepsig201610B (60.37/62)."""
_base_ = ['../../resnetamr/resnetamr_iq-deepsig-201610B.py']
optim_wrapper = dict(optimizer=dict(type='Adam', lr=1e-3))
param_scheduler = dict(
    _delete_=True, type='ReduceOnPlateauParamScheduler', param_name='lr',
    monitor='accuracy/top1', factor=0.5, patience=8, verbose=True,
    rule='greater', min_value=1e-6)
train_cfg = dict(by_epoch=True, max_epochs=500, val_interval=1)
custom_hooks = [dict(type='EarlyStoppingHook', monitor='accuracy/top1',
                     min_delta=0, patience=80, rule='greater')]
