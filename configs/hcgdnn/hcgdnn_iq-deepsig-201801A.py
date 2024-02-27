_base_ = [
    '../_base_/datasets/deepsig/iq-deepsig201801A.py',
    '../_base_/schedules/amc.py',
    '../_base_/runtimes/amc.py'
]

# model settings
model = dict(
    type='SignalClassifier',
    backbone=dict(
        type='HCGDNN',
        num_classes=24,
    ),
    head=dict(
        type='ClsHead',
        loss=dict(
            cnn=dict(type='CrossEntropyLoss', loss_weight=1),
            gru1=dict(type='CrossEntropyLoss', loss_weight=1),
            gru2=dict(type='CrossEntropyLoss', loss_weight=1)
        )
    )
)
