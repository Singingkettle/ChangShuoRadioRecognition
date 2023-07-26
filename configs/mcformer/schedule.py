# optimizer
optimizer = dict(type='Adam', lr=0.002)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='exp', gamma=0.97724)
# set random seed
seed = 0
runner = dict(type='EpochBasedRunner', max_epochs=100)