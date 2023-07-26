_base_ = [
    './schedule.py',
    './default_runtime.py',
    './data_iq-ap-channel-deepsig201610A.py'
]

in_size = 100
out_size = 288
num_classes = 11
model = dict(
    type='FastMLDNN',
    backbone=dict(
        type='FMLNet',
        input_size=in_size,
    ),
    classifier_head=dict(
        type='SSRCNNHead',
        in_size=in_size,
        out_size=out_size,
        num_classes=11,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1,
        ),
        center_head=dict(
            type='CenterHead',
            in_size=out_size,  # keep the same as snr head
            num_classes=11,
            loss_center=dict(
                type='CenterLoss',
                loss_weight=0.003,
            ),
        ),
        init_cfg=[
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        ],
    ),
)

runner = dict(type='EpochBasedRunner', max_epochs=3200)
# Optimizer
# for flops calculation
optimizer = dict(type='Adam', lr=0.00044)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    gamma=0.3,
    step=[800, 1200]
)
