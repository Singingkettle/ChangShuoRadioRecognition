_base_ = [
    '../_base_/datasets/iq-snr-[-8,20]-deepsig-201801A.py',
    '../_base_/schedules/schedule.py',
    '../_base_/default_runtime.py'
]

# Model
model = dict(
    type='SingleHeadClassifier',
    method_name='CNN4-IQ',
    backbone=dict(
        type='CNNNet',
        depth=4,
        in_channels=1,
        out_indices=(3,),
        avg_pool=(1, 8),
    ),
    classifier_head=dict(
        type='ACMHead',
        num_classes=24,
        in_features=10240,
        out_features=128,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
    ),
)
