_base_ = [
    './iq-ap-hisar2019.py',
    '../_base_/schedules/amc.py',
    '../_base_/runtimes/amc.py'
]

# model settings
model = dict(
    type='SignalClassifier',
    backbone=dict(
        type='FastMLDNN',
        num_classes=26,
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)