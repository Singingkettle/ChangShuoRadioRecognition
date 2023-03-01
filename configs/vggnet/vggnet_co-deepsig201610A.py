_base_ = [
    '../_base_/datasets/deepsig/co-deepsig201610A.py',
    '../_base_/schedules/schedule.py',
    '../_base_/default_runtime.py',
]

# Model
model = dict(
    type='VGGNet',
    backbone=dict(
        type='VGGNet'
    ),
    classifier_head=dict(
        type='AMCHead',
        num_classes=11,
        in_size=512,
        out_size=256,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1
        )
    )
)

total_epochs = 800

# optimizer
optimizer = dict(type='Adam', lr=0.00001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')
