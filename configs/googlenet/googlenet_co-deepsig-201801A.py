_base_ = [
    '../_base_/datasets/co-deepsig-201801A.py',
    '../_base_/schedules/schedule.py',
    '../_base_/default_runtime.py'
]

# Model
model = dict(
    type='DNN',
    method_name='GoogleNet-CO',
    backbone=dict(
        type='GoogleNet'
    ),
    classifier_head=dict(
        type='AMCHead',
        num_classes=24,
        in_features=1024,
        out_features=256,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1
        )
    )
)

total_epochs = 100

# optimizer
optimizer = dict(type='Adam', lr=0.00002)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')
