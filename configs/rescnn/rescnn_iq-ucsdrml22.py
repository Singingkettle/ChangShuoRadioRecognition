_base_ = [
    '../_base_/datasets/ucsd/iq-ucsdrml22.py',
    '../_base_/schedules/schedule.py',
    '../_base_/default_runtime.py',
]

# model
model = dict(
    type='ResCNN',
    backbone=dict(
        type='ResCNN',
    ),
    classifier_head=dict(
        type='AMCHead',
        num_classes=11,
        in_size=10240,
        out_size=128,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
    ),
)

# optimizer
optimizer = dict(type='Adam')
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1)