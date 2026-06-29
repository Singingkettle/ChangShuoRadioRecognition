# schedule settings
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.001)
)

# Cosine LR decay over the (capped) epoch budget. The learning rate therefore
# anneals visibly every epoch toward ``eta_min``.
#
# This replaces the previous ``ReduceOnPlateauParamScheduler``: on the large
# RML2018.01A dataset (24 classes, ~3195 train iters/epoch) the validation loss
# kept improving marginally, so the plateau scheduler almost never stepped and
# the LR sat at 5.0000e-04 for 80+ epochs. Combined with ``max_epochs=10000``
# that produced a ~106-day ETA. RML2016.10A/10B converge in well under the new
# cap, so their behaviour is unchanged in practice.
param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=150,
    eta_min=1e-6,
)

# Hard cap on the epoch budget. The completed RML2018.01A models reached their
# best validation accuracy at epochs 77-193, so 150 (plus the EarlyStoppingHook
# in runtimes/amc.py) comfortably covers convergence while bounding the worst
# case to hours instead of months.
train_cfg = dict(by_epoch=True, max_epochs=150, val_interval=1)
val_cfg = dict()
test_cfg = dict()
