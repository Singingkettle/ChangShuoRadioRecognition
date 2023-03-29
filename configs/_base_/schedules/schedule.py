# optimizer
optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')
runner = dict(type='EpochBasedRunner', max_epochs=1200)
