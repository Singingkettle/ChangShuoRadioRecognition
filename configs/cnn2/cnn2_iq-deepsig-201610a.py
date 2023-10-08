_base_ = [
    '../_base_/datasets/deepsig/iq-deepsig201610A.py',
    '../_base_/schedules/amc.py',
    '../_base_/runtimes/amc.py'
]

# model settings
model = dict(
    type='SignalClassifier',
    backbone=dict(
        type='CNN2',
        num_classes=11,
        init_cfg=[
            dict(type='Kaiming', layer='Linear', mode='fan_in'),
            dict(type='Xavier', layer='Conv2d', distribution='uniform'),
        ],
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)
