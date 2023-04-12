_base_ = [
    '../_base_/datasets/ucsd/co-ucsdrml22.py',
    '../_base_/schedules/schedule.py',
    '../_base_/default_runtime.py'
]

# Model
model = dict(
    type='AlexNet',
    backbone=dict(
        type='AlexNet'
    ),
    classifier_head=dict(
        type='AMCHead',
        num_classes=24,
        in_size=256,
        out_size=256,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1
        )
    )
)

# optimizer
optimizer = dict(type='Adam', lr=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1)
