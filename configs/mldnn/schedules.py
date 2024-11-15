# schedule settings
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.0004)
)

train_cfg = dict(by_epoch=True, max_epochs=400, val_interval=1)  # train 5 epochs
val_cfg = dict()
test_cfg = dict()
randomness=dict(seed=3407)