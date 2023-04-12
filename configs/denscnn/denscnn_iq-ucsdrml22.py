_base_ = [
    '../_base_/datasets/ucsd/iq-ucsdrml22.py',
    '../_base_/schedules/schedule.py',
    '../_base_/default_runtime.py'
]

# model
model = dict(
    type='DensCNN',
    backbone=dict(
        type='DensCNN',
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

checkpoint_config = dict(interval=1)
evaluation = dict(interval=1)