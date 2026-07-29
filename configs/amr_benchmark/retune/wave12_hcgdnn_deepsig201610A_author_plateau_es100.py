"""Wave-12 Tier-A: HCGDNN author-exact recipe (upstream published release).

Recovered from origin/main: Adam 4.4e-4 + ReduceOnPlateau on fused val top-1
(factor 0.3, patience 30, min lr 1e-7), EarlyStopping min_delta=0 patience=100,
HCGDNNHook fusion weights, batch 640, no IQ L2. All our prior waves replaced
this with MultiStep/Cosine + tight ES and never ran the adaptive-plateau
schedule the paper numbers came from. max_epochs capped at 2500 for sanity
(author file said 10000; ES patience 100 stops it long before).
"""
_base_ = ['../../hcgdnn/hcgdnn_iq-deepsig-201610A.py']

param_scheduler = dict(
    _delete_=True,
    type='ReduceOnPlateauParamScheduler',
    param_name='lr',
    monitor='accuracy/top1',
    factor=0.3,
    patience=30,
    verbose=True,
    rule='greater',
    min_value=1e-7,
)

train_cfg = dict(by_epoch=True, max_epochs=2500, val_interval=1)

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='accuracy/top1', min_delta=0,
         patience=100, rule='greater'),
    dict(type='HCGDNNHook'),
]
