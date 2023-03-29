_base_ = [
    '../_base_/datasets/doctorhe/co-doctorhe-case2-sqcd-s6.py',
    '../_base_/schedules/schedule.py',
    '../_base_/default_runtime.py',
]

# Model
model = dict(
    type='AlexNet',
    backbone=dict(
        type='AlexNet'
    ),
    classifier_head=dict(
        type='AMCHead',
        num_classes=6,
        in_size=256,
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
