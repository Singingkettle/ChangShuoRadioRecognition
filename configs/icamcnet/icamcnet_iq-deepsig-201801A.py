_base_ = [
    '../_base_/datasets/deepsig/iq-deepsig201801A.py',
    '../_base_/schedules/amc.py',
    '../_base_/runtimes/amc.py'
]

# model settings
model = dict(
    type='SignalClassifier',
    backbone=dict(
        type='ICAMCNet',
        frame_length=1024,
        num_classes=24,
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)
