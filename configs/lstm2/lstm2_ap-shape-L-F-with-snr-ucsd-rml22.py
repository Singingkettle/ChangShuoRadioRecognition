_base_ = [
    '../_base_/datasets/ucsd/ap-shape-L-F-with-snr-ucsdrml22.py',
    './schedules.py',
    './runtimes.py'
]

hidden_size = 128
# model settings
model = dict(
    type='SignalClassifier',
    backbone=dict(
        type='LSTM2',
        hidden_size=hidden_size,
    ),
    head=dict(
        type='SNRAuxiliaryHead',
        num_classes=10,
        num_snr=21,
        input_size=hidden_size,
        is_shallow=True,
        loss_fcls=dict(type='CrossEntropyLoss', loss_weight=1.0),
        loss_cls=dict(type='CrossEntropyLoss', loss_weight=0.1),
        loss_snr=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)
