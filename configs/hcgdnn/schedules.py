# schedule settings
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.00044)
)

# Cosine LR decay over the (capped) epoch budget so the learning rate anneals
# visibly every epoch toward ``eta_min``.
#
# Previously this used ``ReduceOnPlateauParamScheduler`` (monitor
# ``accuracy/top1``, patience 30) with ``max_epochs=10000``. Because the fused
# validation accuracy keeps inching up by tiny amounts, the plateau scheduler
# almost never stepped, so the LR sat frozen at 4.4000e-04 while training ran to
# epoch 200+ (observed ~5-day ETA). Switching to ``CosineAnnealingLR`` guarantees
# monotonic decay, and the EarlyStoppingHook in ``runtimes.py`` (min_delta 0.1pp,
# patience 15) stops training once validation accuracy genuinely plateaus. The
# CheckpointHook's ``save_best='accuracy/top1'`` keeps the best-val checkpoint for
# testing. Matches the shared baseline recipe (commit e9c3c99) while preserving
# HCGDNN's original optimizer LR.
param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=150,
    eta_min=1e-6,
)

train_cfg = dict(by_epoch=True, max_epochs=150, val_interval=1)
val_cfg = dict()
test_cfg = dict()
