# schedule settings
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.00044)
)

param_scheduler = dict(
    type='ReduceOnPlateauParamScheduler',
    param_name='lr',
    monitor='accuracy/top1',
    factor=0.3,
    patience=30,
    verbose=True,
    rule='greater',
    min_value=0.0000001,
)

train_cfg = dict(by_epoch=True, max_epochs=10000, val_interval=1)  # train 5 epochs
val_cfg = dict()
test_cfg = dict()
