_base_ = [
    './schedule.py',
    './default_runtime.py',
    './data_iq-ap-channel-deepsig201610A.py'
]

in_size = 100
out_size = 256
num_classes = 11
model = dict(
    type='FastMLDNN',
    backbone=dict(
        type='FMLNet',
        in_size=4,
        channel_mode=True,
        skip_connection=False,
    ),
    classifier_head=dict(
        type='AMCHead',
        in_size=in_size,
        out_size=out_size,
        num_classes=num_classes,
        loss_cls=dict(
            type='FocalLoss',
            loss_weight=1,
            alpha=0.5,
        ),
    ),
)

# for flops calculation
input_shape = [(2, 1, 128), (2, 1, 128), (1, 128, 128)]
