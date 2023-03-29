_base_ = [
    './schedule.py',
    '../_base_/default_runtime.py',
    './data_iq-ap-channel-deepsig201610A.py'
]

in_size = 100
out_size = 288
num_classes = 11
model = dict(
    type='FastMLDNN',
    backbone=dict(
        type='FMLNetV4',
        input_size=in_size,
    ),
    classifier_head=dict(
        type='AMCHead',
        num_classes=11,
        in_size=in_size,
        out_size=out_size,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1
        )
    )
)

# for flops calculation
optimizer = dict(type='Adam', lr=0.00044)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    gamma=0.3,
    step=[800, 1200]
)
