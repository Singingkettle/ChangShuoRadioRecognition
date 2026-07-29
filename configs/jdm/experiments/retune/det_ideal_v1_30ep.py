# Ideal-protocol detector train: CSRD v1 only (no random impairments / infdB).
# 30-ep Adam 1e-3 — paper Sec. VI schedule length; fair Fig. 8(a) ideal comparison.
_base_ = '../jdm-det_fft-csrd_anchor096146_bw20_5ep.py'

_ideal_versions = ['v1']

train_dataloader = dict(dataset=dict(versions=_ideal_versions))
val_dataloader = dict(dataset=dict(versions=_ideal_versions))
test_dataloader = dict(dataset=dict(versions=_ideal_versions))

param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=30,
    eta_min=1e-6,
)

train_cfg = dict(by_epoch=True, max_epochs=30, val_interval=1)
custom_hooks = []

work_dir = 'work_dirs/jdm/retune/det_ideal_v1_30ep'
