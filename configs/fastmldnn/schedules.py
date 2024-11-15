# schedule settings
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.00044)
)

param_scheduler = dict(
    type='MultiStepLR',
    by_epoch=True,
    milestones=[800, 1200],
    gamma=0.3
)

train_cfg = dict(by_epoch=True, max_epochs=3200, val_interval=1)
val_cfg = dict()
test_cfg = dict()
