# schedule settings
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.00044)
)

# Cosine LR decay over the (capped) epoch budget so the learning rate anneals
# visibly every epoch toward ``eta_min``.
#
# Previously this used ``MultiStepLR(milestones=[800, 1200])`` with
# ``max_epochs=3200``: the first LR drop would not occur until epoch 800, far
# beyond any sane convergence point, so in practice the LR never decayed and
# training never stopped. Switching to ``CosineAnnealingLR`` makes the LR decay
# every epoch; the EarlyStoppingHook added to ``runtimes.py`` (min_delta 0.1pp,
# patience 15) stops training at the validation-accuracy plateau, and the
# CheckpointHook's ``save_best='accuracy/top1'`` keeps the best-val checkpoint
# for testing. Matches the shared baseline recipe (commit e9c3c99) while
# preserving FastMLDNN's original optimizer LR. Only
# ``fastmldnn_iq-ap-deepsig-201610A.py`` consumes this file; the other three
# FastMLDNN configs already inherit ``_base_/schedules/amc.py``.
param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=150,
    eta_min=1e-6,
)

train_cfg = dict(by_epoch=True, max_epochs=150, val_interval=1)
val_cfg = dict()
test_cfg = dict()
