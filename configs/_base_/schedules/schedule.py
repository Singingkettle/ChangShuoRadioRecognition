# Optimizer
optimizer = dict(type='Adam')
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')
# set random seed
seed = 0
runner = dict(type='EpochBasedRunner', max_epochs=400)