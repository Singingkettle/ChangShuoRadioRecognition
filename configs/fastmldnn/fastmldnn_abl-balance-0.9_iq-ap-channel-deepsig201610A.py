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
        dp=0.07,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/home/citybuster/Projects/ChangShuoRadioRecognition/work_dirs/fastmldnn_iq-ap-channel-deepsig201610A/init/init.pth',
            prefix='backbone'
        )
    ),
    classifier_head=dict(
        type='FastMLDNNHead',
        num_classes=11,
        in_size=in_size,
        out_size=out_size,
        balance=0.9,
        loss_cls=dict(
            type='FocalLoss',
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
optimizer = dict(type='Adam', lr=0.0001054)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    gamma=0.3,
    step=[20, 80, 400, 600, 760]
)
evaluation = dict(interval=1)
checkpoint_config = dict(interval=1)
seed = 0
