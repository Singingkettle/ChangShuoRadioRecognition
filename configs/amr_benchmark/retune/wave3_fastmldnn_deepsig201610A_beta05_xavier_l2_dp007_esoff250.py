"""Wave-3 retune: FastMLDNN @ RML2016.10A — Wave-2 combo + 250 epochs, ES off."""

_base_ = [
    './wave2_fastmldnn_deepsig201610A_beta05_xavier_l2_dp007_esoff.py',
]

train_cfg = dict(by_epoch=True, max_epochs=250, val_interval=1)

param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=250,
    eta_min=1e-6,
)
