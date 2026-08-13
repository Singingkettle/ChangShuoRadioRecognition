# Efficient M-block CNN (MCNet) for robust AMC on I/Q.
# Paper: "MCNet: An Efficient CNN Architecture for Robust Automatic Modulation Classification", IEEE Commun. Lett. (2020).
_base_ = [
    '../_base_/datasets/deepsig/iq-deepsig201610A.py',
    '../_base_/schedules/amc.py',
    '../_base_/runtimes/amc.py'
]

# model settings
model = dict(
    type='SignalClassifier',
    backbone=dict(
        type='MCNet',
        num_classes=11,
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)