total_epochs = 800
# optimizer
optimizer = dict(type='Adam', lr=0.002)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')
