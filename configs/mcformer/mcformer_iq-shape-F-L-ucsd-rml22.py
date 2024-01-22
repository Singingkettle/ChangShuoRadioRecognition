_base_ = [
    '../_base_/datasets/ucsd/iq-shape-F-L-ucsdrml22.py',
    '../_base_/schedules/amc.py',
    '../_base_/runtimes/amc.py'
]

# Model
model = dict(
    type='SignalClassifier',
    backbone=dict(
        type='MCformer',
        fea_dim=32,
        num_classes=10,
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)

