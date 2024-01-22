_base_ = [
    './ap-deepsig201604C.py',
    '../_base_/schedules/amc.py',
    '../_base_/runtimes/amc.py'
]

# model settings
model = dict(
    type='SignalClassifier',
    backbone=dict(
        type='DAE',
        num_classes=11,
    ),
    head=dict(
        type='DAEHead',
        loss_cls=dict(type='CrossEntropyLoss', loss_weight=0.1),
        loss_mse=dict(type='MSELoss', loss_weight=0.9, reduction='mean', _scope_='mmdet')
    )
)
