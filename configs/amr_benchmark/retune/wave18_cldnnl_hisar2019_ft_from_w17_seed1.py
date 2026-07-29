"""Wave-18 Tier-B: Hisar FT from wave-17 Keras-recipe best (70.263 vs 75).

Half LR (5e-4) + plateau + ES, warm-started from the wave-17 checkpoint that
moved the needle. Same ratchet that worked for HCGDNN Tier-A.
"""
_base_ = ['../../cldnnl/cldnnl_iq-hisar-2019.py']

load_from = 'work_dirs/amr_benchmark_retune/cldnnl/hisar2019/amrb_plateau_w17/best_accuracy_top1_epoch_148.pth'

optim_wrapper = dict(optimizer=dict(type='Adam', lr=5e-4))

param_scheduler = dict(
    _delete_=True,
    type='ReduceOnPlateauParamScheduler',
    param_name='lr',
    monitor='accuracy/top1',
    factor=0.5,
    patience=5,
    verbose=True,
    rule='greater',
    min_value=1e-6,
)

train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=1)

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='accuracy/top1', min_delta=0,
         patience=40, rule='greater'),
]

randomness = dict(seed=1)
