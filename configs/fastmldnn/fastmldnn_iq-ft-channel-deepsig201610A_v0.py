_base_ = [
    './schedule.py',
    './default_runtime.py',
    './data_iq-ft-channel-deepsig201610A.py'
]

in_size = 100
out_size = 288
num_classes = 11
model = dict(
    type='FastMLDNN',
    backbone=dict(
        type='FMLNetV3',
        input_size=in_size,
    ),
    classifier_head=dict(
        type='FastMLDNNHeadV2',
        num_classes=11,
        levels=(256, 256, 100),
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

runner = dict(type='EpochBasedRunner', max_epochs=400)
# Optimizer
# for flops calculation
optimizer = dict(type='Adam', lr=0.00044)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    gamma=0.3,
    step=[100, 300]
)
