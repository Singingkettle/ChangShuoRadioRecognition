"""Wave-19: CLDNNL Hisar round-2 FT from w18 seed0 (val 70.81 → target 75)."""
_base_ = ['../../cldnnl/cldnnl_iq-hisar-2019.py']
load_from = (
    'work_dirs/amr_benchmark_retune/cldnnl/hisar2019/'
    'ft_from_w17_seed0_w18/best_accuracy_top1_epoch_65.pth')
optim_wrapper = dict(optimizer=dict(type='Adam', lr=2.5e-4))
param_scheduler = dict(
    _delete_=True, type='ReduceOnPlateauParamScheduler', param_name='lr',
    monitor='accuracy/top1', factor=0.5, patience=5, verbose=True,
    rule='greater', min_value=1e-6)
train_cfg = dict(by_epoch=True, max_epochs=250, val_interval=1)
custom_hooks = [dict(type='EarlyStoppingHook', monitor='accuracy/top1',
                     min_delta=0, patience=35, rule='greater')]
randomness = dict(seed=4)
