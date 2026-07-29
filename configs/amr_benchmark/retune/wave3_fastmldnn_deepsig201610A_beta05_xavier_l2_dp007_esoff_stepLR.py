"""Wave-3 retune: FastMLDNN @ RML2016.10A — Wave-2 combo + StepLR decay.

Paper Keras lr_config drops ×0.1 @ epochs 100 and 150; cosine-only may
under-shoot peak SNR on the winning Wave-2 stack.
"""

_base_ = [
    './wave2_fastmldnn_deepsig201610A_beta05_xavier_l2_dp007_esoff.py',
]

train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=1)

# ``_delete_=True`` drops inherited CosineAnnealingLR keys (T_max, eta_min) that
# MMEngine would otherwise merge in and pass to MultiStepLR.
param_scheduler = dict(
    _delete_=True,
    type='MultiStepLR',
    by_epoch=True,
    milestones=[100, 150],
    gamma=0.1,
)
