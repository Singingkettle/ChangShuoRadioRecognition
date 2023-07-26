_base_ = [
    './data_iq-deepsig201610A.py',
    './schedule.py',
    '../_base_/default_runtime.py'
]

# Model
model = dict(
    type='MCformer',
    method_name='MCformerSmall',
    backbone=dict(
        type='MCformerNet',
        dim=8
    ),
    classifier_head=dict(
        type='AMCHead',
        num_classes=11,
        in_size=32,
        out_size=128,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
    ),
)
