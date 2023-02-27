_base_ = [
    '../_base_/datasets/iq-deepsig201610A.py',
    '../_base_/schedules/schedule.py',
    '../_base_/default_runtime.py'
]

# Model
model = dict(
    type='SingleHeadClassifier',
    method_name='CNN3-IQ',
    backbone=dict(
        type='CNNNet',
        depth=3,
        in_channels=1,
        out_indices=(2,),
    ),
    classifier_head=dict(
        type='ACMHead',
        num_classes=11,
        in_features=10720,
        out_features=256,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
    ),
)
