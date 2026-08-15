# PET rotation + compact CGDNN for AMC on I/Q.
# Paper: "An Efficient Deep Learning Model for Automatic Modulation Recognition Based on Parameter Estimation and Transformation", IEEE Commun. Lett. (2021).
_base_ = [
    '../_base_/datasets/deepsig/iq-shape-L-F-deepsig201610A.py',
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