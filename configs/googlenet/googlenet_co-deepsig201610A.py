_base_ = [
    '../_base_/datasets/co-deepsig201610A.py',
    '../_base_/schedules/schedule.py',
    '../_base_/default_runtime.py',
]

# Model
model = dict(
    type='SingleHeadClassifier',
    method_name='GoogleNet-CO',
    backbone=dict(
        type='GoogleNet'
    ),
    classifier_head=dict(
        type='ACMHead',
        num_classes=11,
        in_features=1024,
        out_features=256,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1
        )
    )
)

total_epochs = 800

# optimizer
optimizer = dict(type='Adam', lr=0.00001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')
