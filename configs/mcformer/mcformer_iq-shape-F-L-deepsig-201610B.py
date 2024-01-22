_base_ = [
    '../_base_/datasets/deepsig/iq-shape-F-L-deepsig201610B.py',
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

