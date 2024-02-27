_base_ = [
    './schedule.py',
    '../_base_/default_runtime.py',
    '../_base_/datasets/deepsig/iq-channel-deepsig201610A.py',
]

in_size = 100
out_size = 288
# Model
model = dict(
    type='HCGDNN',
    backbone=dict(
        type='HCGNetGRU1',
        input_size=in_size,
    ),
    classifier_head=dict(
        type='AMCHead',
        in_size=in_size,
        out_size=out_size,
        num_classes=11,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1,
        ),
    ),
)
