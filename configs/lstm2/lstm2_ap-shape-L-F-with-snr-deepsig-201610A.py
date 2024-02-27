_base_ = [
    '../_base_/datasets/deepsig/ap-shape-L-F-with-snr-deepsig201610A.py',
    './schedules.py',
    '../_base_/runtimes/amc.py'
]

hidden_size = 128
# model settings
model = dict(
    type='SignalClassifier',
    backbone=dict(
        type='LSTM2',
        hidden_size=hidden_size,
        init_cfg=dict(
            type='LSTM', gain=1, layer='LSTM'
        )
    ),
    head=dict(
        type='SNRAuxiliaryHead',
        num_classes=11,
        num_snr=20,
        input_size=hidden_size,
        output_size=256,
        snr_output_size=256,
        loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0),
        loss_snr=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)
