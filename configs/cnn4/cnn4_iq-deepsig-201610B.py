_base_ = [
    '../_base_/datasets/deepsig/iq-deepsig201610B.py',
    '../_base_/schedules/amc.py',
    '../_base_/runtimes/amc.py'
]

# model settings
model = dict(
    type='SignalClassifier',
    backbone=dict(
        type='CNN4',
        num_classes=10,
        init_cfg=dict(type='Xavier', layer='Conv2d')
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)
