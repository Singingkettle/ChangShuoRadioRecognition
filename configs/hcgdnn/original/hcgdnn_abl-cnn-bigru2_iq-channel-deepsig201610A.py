_base_ = [
    './schedule.py',
    './data_iq-channel-deepsig201610A.py',
    '../_base_/default_runtime.py',
]

in_size = 100
out_size = 288
heads = ['CNN', 'BiGRU2']
# Model
model = dict(
    type='HCGDNN',
    backbone=dict(
        type='HCGNet',
        heads=heads,
        input_size=in_size,
    ),
    classifier_head=dict(
        type='HCGDNNHead',
        in_size=in_size,
        out_size=out_size,
        num_classes=11,
        heads=heads,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1,
        ),
    ),
)
