_base_ = [
    '../_base_/datasets/ucsd/iq-ucsdrml22.py',
    '../_base_/schedules/schedule.py',
    '../_base_/default_runtime.py'
]

# Model
model = dict(
    type='CNN4',
    backbone=dict(
        type='CNNNet',
        depth=4,
        in_channels=1,
        out_indices=(3,),
    ),
    classifier_head=dict(
        type='AMCHead',
        num_classes=10,
        in_size=10880,
        out_size=128,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
    ),
)
optimizer = dict(type='Adam', lr=0.0001)
evaluation = dict(interval=1)