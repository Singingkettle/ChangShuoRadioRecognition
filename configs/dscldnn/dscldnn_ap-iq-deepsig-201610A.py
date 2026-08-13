# Dual-stream CNN–LSTM AMC on I/Q and A/P.
# Paper: "Automatic Modulation Classification Using CNN-LSTM Based Dual-Stream Structure", IEEE Access (2020).
_base_ = [
    '../_base_/datasets/deepsig/ap-iq-deepsig201610A.py',
    '../_base_/schedules/amc.py',
    '../_base_/runtimes/amc.py'
]

# model settings
model = dict(
    type='SignalClassifier',
    backbone=dict(
        type='DSCLDNN',
        num_classes=11,
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)