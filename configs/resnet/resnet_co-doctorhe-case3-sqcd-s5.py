_base_ = [
    '../_base_/datasets/doctorhe/co-doctorhe-case3-sqcd-s5.py',
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
        num_classes=6,
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
evaluation = dict(interval=1)
checkpoint_config = dict(interval=1)
runner = dict(type='EpochBasedRunner', max_epochs=100)
