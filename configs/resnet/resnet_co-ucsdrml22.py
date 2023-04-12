_base_ = [
    '../_base_/datasets/ucsd/co-ucsdrml22.py',
    '../_base_/schedules/schedule.py',
    '../_base_/default_runtime.py',
]

# Model
model = dict(
    type='ResNet',
    backbone=dict(
        type='ResNet'
    ),
    classifier_head=dict(
        type='AMCHead',
        num_classes=11,
        in_size=2048,
        out_size=256,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1
        )
    )
)


# optimizer
optimizer = dict(type='Adam', lr=0.00001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1)
