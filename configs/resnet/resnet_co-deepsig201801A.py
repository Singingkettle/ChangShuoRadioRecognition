_base_ = [
    '../_base_/datasets/co-deepsig-201801A.py',
    '../_base_/schedules/schedule.py',
    '../_base_/default_runtime.py',
]

# Model
model = dict(
    type='SingleHeadClassifier',
    method_name='ResNet-CO',
    backbone=dict(
        type='ResNet'
    ),
    classifier_head=dict(
        type='ACMHead',
        num_classes=24,
        in_features=2048,
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
