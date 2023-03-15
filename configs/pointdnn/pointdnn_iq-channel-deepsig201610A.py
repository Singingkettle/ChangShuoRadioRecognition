_base_ = [
    './schedule.py',
    './data_iq-channel-deepsig201610A.py',
    '../_base_/default_runtime.py',
]

# Model
model = dict(
    type='PointDNN',
    backbone=dict(
        type='COCNetTiny',
    ),
    classifier_head=dict(
        type='DSCLDNNHead',
        num_classes=11,
        in_size=320,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
    ),
)
