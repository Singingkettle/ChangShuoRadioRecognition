"""Wave-15 Tier-B: AMR-Benchmark ORIGINAL (Keras upstream) training recipe.

mcnet @ deepsig201610A: measured 55.98/58 gap -2.02. Waves 1-9 tried generic tweaks (ES patience,
lr2e4+warmup, xavier); wave-12 showed the author's published schedule is the
lever that actually moves results (HCGDNN plateau recipe -> new best). The
upstream AMR-Benchmark Keras training loop is: Adam lr=1e-3 +
ReduceLROnPlateau(factor 0.5, patience 5) + EarlyStopping(patience 50) with
per-epoch val — never tried on the Tier-B ports.
"""
_base_ = ['../../mcnet/mcnet_iq-deepsig-201610A.py']

optim_wrapper = dict(optimizer=dict(type='Adam', lr=1e-3))

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

train_cfg = dict(by_epoch=True, max_epochs=400, val_interval=1)

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='accuracy/top1', min_delta=0,
         patience=50, rule='greater'),
]
