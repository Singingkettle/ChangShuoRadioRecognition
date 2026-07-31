"""Wave-28: DAE@Hisar polish — FT the baseline last ckpt at LR 1e-4.

Fresh plateau run regressed (peak 57.47 vs baseline 61.39, pass 69.0);
ratchet from the baseline model instead.
"""
_base_ = ['../../dae/dae_ap-hisar-2019.py']
load_from = 'work_dirs/amr_benchmark/dae/hisar2019/epoch_85.pth'
optim_wrapper = dict(optimizer=dict(type='Adam', lr=1e-4))
param_scheduler = dict(
    _delete_=True, type='ReduceOnPlateauParamScheduler', param_name='lr',
    monitor='accuracy/top1', factor=0.5, patience=6, verbose=True,
    rule='greater', min_value=1e-6)
train_cfg = dict(by_epoch=True, max_epochs=120, val_interval=1)
custom_hooks = [dict(type='EarlyStoppingHook', monitor='accuracy/top1',
                     min_delta=0, patience=30, rule='greater')]
