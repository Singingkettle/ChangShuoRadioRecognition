_base_ = [
    '../_base_/datasets/deepsig/iq-deepsig201801A.py',
    '../_base_/schedules/schedule.py',
    '../_base_/default_runtime.py'
]

# model
model = dict(
    type='ResCNN',
    backbone=dict(
        type='ResCNN',
        avg_pool=(1, 8),
    ),
    classifier_head=dict(
        type='AMCHead',
        num_classes=24,
        in_size=10240,
        out_size=128,
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
