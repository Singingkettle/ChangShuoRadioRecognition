# Multi-channel CNN + LSTM (MCLDNN) on I/Q.
# Paper: "A Spatiotemporal Multi-Channel Learning Framework for Automatic Modulation Recognition", IEEE WCL (2020).
_base_ = [
    '../_base_/datasets/deepsig/iq-l2norm-deepsig201610A.py',
    '../_base_/schedules/amc.py',
    '../_base_/runtimes/amc.py'
]

# model settings
model = dict(
    type='SignalClassifier',
    backbone=dict(
        type='MCLDNN',
        num_classes=11,
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)