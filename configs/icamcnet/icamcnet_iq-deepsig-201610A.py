# IC-AMCNet deep CNN with Gaussian noise regularization on I/Q.
# Paper: "CNN-Based Automatic Modulation Classification for Beyond 5G Communications", IEEE Commun. Lett. (2020).
_base_ = [
    '../_base_/datasets/deepsig/iq-deepsig201610A.py',
    '../_base_/schedules/amc.py',
    '../_base_/runtimes/amc.py'
]

# model settings
model = dict(
    type='SignalClassifier',
    backbone=dict(
        type='ICAMCNet',
        num_classes=11,
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)