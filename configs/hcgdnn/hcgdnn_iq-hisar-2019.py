_base_ = [
    './iq-hisar2019.py',
    './runtimes.py',
    './schedules.py'
]

# model settings
model = dict(
    type='SignalClassifier',
    backbone=dict(
        type='HCGDNN',
        num_classes=26,
    ),
    head=dict(
        type='HCGDNNHead',
        loss=dict(
            cnn=dict(type='CrossEntropyLoss', loss_weight=1),
            gru1=dict(type='CrossEntropyLoss', loss_weight=1),
            gru2=dict(type='CrossEntropyLoss', loss_weight=1)
        )
    )
)
