# schedule settings
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.0004)
)

# param_scheduler = dict(
#     type='ReduceOnPlateauParamScheduler',
#     param_name='lr',
#     monitor='loss/classification',
#     factor=0.4,
#     patience=5,
#     verbose=True,
#     min_value=0.0000001,
# )

train_cfg = dict(by_epoch=True, max_epochs=400, val_interval=1)  # train 5 epochs
val_cfg = dict()
test_cfg = dict()
