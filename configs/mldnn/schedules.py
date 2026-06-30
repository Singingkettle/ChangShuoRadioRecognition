# schedule settings
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.0004)
)

# Cosine LR decay over the (capped) epoch budget so the learning rate anneals
# visibly every epoch toward ``eta_min``.
#
# Previously this file declared *no* ``param_scheduler`` at all, so the LR was
# frozen at 4.0000e-04 for the entire run, and ``max_epochs=400`` let large
# datasets (RML2018.01A: ~2000 train iters/epoch) drift for multiple days. The
# EarlyStoppingHook in ``_base_/runtimes/amc.py`` (monitor ``accuracy/top1``,
# min_delta 0.1pp, patience 15) now terminates training once the validation
# accuracy genuinely plateaus, and the best-val checkpoint is the one tested.
# Matches the shared baseline recipe fixed in ``_base_/schedules/amc.py``
# (commit e9c3c99) while preserving MLDNN's original optimizer LR.
param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=150,
    eta_min=1e-6,
)

train_cfg = dict(by_epoch=True, max_epochs=150, val_interval=1)
val_cfg = dict()
test_cfg = dict()
