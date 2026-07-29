"""Wave-3 retune: FastMLDNN @ RML2016.10A — Wave-2 combo + 200 epochs, ES off.

Val acc still climbing at epoch 115–141 in Wave-2 best; paper uses 400 ep.
"""

_base_ = [
    './wave2_fastmldnn_deepsig201610A_beta05_xavier_l2_dp007_esoff.py',
]

train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=1)

param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=200,
    eta_min=1e-6,
)
