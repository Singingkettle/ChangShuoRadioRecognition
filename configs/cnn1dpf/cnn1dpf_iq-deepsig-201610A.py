# Parallel-fusion 1-D CNN (A/P branches) for AMC.
# Paper: "Automatic Modulation Classification Using Parallel Fusion of Convolutional Neural Networks".
_base_ = [
    '../_base_/datasets/deepsig/ap-deepsig201610A.py',
    '../_base_/schedules/amc.py',
    '../_base_/runtimes/amc.py'
]

# model settings
model = dict(
    type='SignalClassifier',
    backbone=dict(
        type='CNN1DPF',
        num_classes=11,
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)
