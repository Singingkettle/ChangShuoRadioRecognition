"""Wave-3 retune: FastMLDNN @ RML2016.10A — Wave-2 combo + 300 epochs, ES off.

Siege round 3: extend winning esoff250 recipe toward paper 400-ep budget.
"""

_base_ = [
    './wave2_fastmldnn_deepsig201610A_beta05_xavier_l2_dp007_esoff.py',
]

train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=1)

param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=300,
    eta_min=1e-6,
)
