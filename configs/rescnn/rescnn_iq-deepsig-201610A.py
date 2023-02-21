_base_ = [
    '../_base_/datasets/iq-deepsig-201610A.py',
    '../_base_/schedules/schedule.py',
    '../_base_/default_runtime.py',
]

# model
model = dict(
    type='SingleHeadClassifier',
    method_name='ResCNN-IQ',
    backbone=dict(
        type='ResCNN',
    ),
    classifier_head=dict(
        type='ACMHead',
        num_classes=11,
        in_features=10240,
        out_features=128,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
    ),
)

# optimizer
optimizer = dict(type='Adam')
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')
