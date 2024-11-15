_base_ = [
    './iq-ap-deepsig201610A.py',
    './runtimes.py',
    './schedules.py'
]

# model settings
model = dict(
    type='SignalClassifier',
    backbone=dict(
        type='FastMLDNN',
        num_classes=11,
    ),
    head=dict(
        type='FastMLDNNHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        beta=0,
    )
)
