_base_ = [
    '../_base_/datasets/deepsig/iq-amrb-deepsig201610A.py',
    '../_base_/schedules/schedule.py',
    '../_base_/default_runtime.py',
]

drop_rate = 0.6
# model
model = dict(
    type='ResCNN',
    backbone=dict(
        type='AMRBResCNN',
        drop_rate=drop_rate,
    ),
    classifier_head=dict(
        type='AMCHead',
        num_classes=11,
        in_size=20480,
        out_size=128,
        drop_rate=drop_rate,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
    ),
)

total_epochs = 1000
# optimizer
optimizer = dict(type='Adam')
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    gamma=0.5,
    step=[800])
