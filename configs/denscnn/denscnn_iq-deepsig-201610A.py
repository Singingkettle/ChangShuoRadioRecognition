_base_ = [
    '../_base_/datasets/iq-deepsig-201610A.py',
    '../_base_/schedules/schedule.py',
    '../_base_/default_runtime.py'
]

# model
model = dict(
    type='DNN',
    method_name='DensCNN-IQ',
    backbone=dict(
        type='DensCNN',
    ),
    classifier_head=dict(
        type='ClassificationHead',
        num_classes=11,
        in_features=10240,
        out_features=128,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
    ),
)
