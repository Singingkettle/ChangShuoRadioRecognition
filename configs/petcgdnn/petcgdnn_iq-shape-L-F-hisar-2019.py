_base_ = [
    '../_base_/datasets/hisar/iq-shape-L-F-hisar2019.py',
    '../_base_/schedules/amc.py',
    '../_base_/runtimes/amc.py'
]

# model settings
model = dict(
    type='SignalClassifier',
    backbone=dict(
        type='PETCGDNN',
        frame_length=1024,
        num_classes=26,
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)
