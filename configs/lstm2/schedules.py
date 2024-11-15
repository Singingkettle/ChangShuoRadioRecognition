# schedule settings
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.001)
)

param_scheduler = dict(
    type='MultiStepLR',
    by_epoch=True,
    milestones=[800, 1200],
    gamma=0.3
)

train_cfg = dict(by_epoch=True, max_epochs=10000, val_interval=1)  # train 5 epochs
val_cfg = dict()
test_cfg = dict()
