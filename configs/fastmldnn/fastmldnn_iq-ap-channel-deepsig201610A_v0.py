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
    init_cfg=dict(
        type='Pretrained',
        checkpoint='/home/citybuster/Projects/ChangShuoRadioRecognition/work_dirs/fastmldnn_iq-ap-channel-deepsig201610A_v1/epoch_757.pth',
        prefix='backbone'
    ),
    backbone=dict(
        type='FMLNet',
        input_size=in_size,
        dp=0.07,
    ),
    classifier_head=dict(
        type='FastMLDNNHead',
        num_classes=11,
        in_size=in_size,
        out_size=out_size,
        balance=0.5,
        loss_cls=dict(
            type='FocalLoss',
            loss_weight=1,
        ),
    )
)

runner = dict(type='EpochBasedRunner', max_epochs=3200)
# Optimizer
# for flops calculation
optimizer = dict(type='Adam', lr=0.00001054)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='fixed',
)
evaluation = dict(interval=1)
checkpoint_config = dict(interval=1)
seed = 0
