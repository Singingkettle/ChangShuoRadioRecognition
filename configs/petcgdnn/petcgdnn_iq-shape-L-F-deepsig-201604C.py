_base_ = [
    '../_base_/datasets/deepsig/iq-shape-L-F-deepsig201604C.py',
    '../_base_/schedules/amc.py',
    '../_base_/runtimes/amc.py'
]

# model settings
model = dict(
    type='SignalClassifier',
    backbone=dict(
        type='PETCGDNN',
        num_classes=11,
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)
