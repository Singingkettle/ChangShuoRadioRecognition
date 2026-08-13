# Two-layer LSTM AMC on amplitude/phase (L×F).
# Paper: "Deep Learning Models for Wireless Signal Classification With Distributed Low-Cost Spectrum Sensors", IEEE TCCN (2018).
_base_ = [
    '../_base_/datasets/deepsig/ap-shape-L-F-deepsig201610A.py',
    '../_base_/schedules/amc.py',
    '../_base_/runtimes/amc.py'
]

# model settings
model = dict(
    type='SignalClassifier',
    backbone=dict(
        type='LSTM2',
        num_classes=11,
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)