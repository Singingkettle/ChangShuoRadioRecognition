_base_ = [
    './data_second-stage-csrr2023.py',
    '../_base_/schedules/schedule.py',
    '../_base_/default_runtime.py'
]

in_size = 100
out_size = 288
model = dict(
    type='CGDNN2',
    backbone=dict(
        type='HCGNetGRU2',
        input_size=in_size,
    ),
    classifier_head=dict(
        type='AMCHead',
        num_classes=6,
        in_size=in_size,
        out_size=out_size,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1
        ),
        init_cfg=[
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        ],
    )
)

runner = dict(type='EpochBasedRunner', max_epochs=100)
# Optimizer
# for flops calculation
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.00005)
# learning policy
lr_config = dict(policy='fixed')
evaluation = dict(interval=1)
checkpoint_config = dict(interval=1)
