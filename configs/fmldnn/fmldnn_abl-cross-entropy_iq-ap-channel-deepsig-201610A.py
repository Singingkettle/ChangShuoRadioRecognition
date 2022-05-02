_base_ = [
    './schedule.py',
    '../_base_/default_runtime.py',
    './data_iq-ap-channel-deepsig-201610A.py'
]

in_features = 100
out_features = 256
num_classes = 11
model = dict(
    type='DNN',
    method_name='Fast MLDNN-V2',
    backbone=dict(
        type='FMLNet',
        in_features=4,
        channel_mode=True,
        skip_connection=False,
    ),
    classifier_head=dict(
        type='AMCHead',
        in_features=in_features,
        out_features=out_features,
        num_classes=num_classes,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1,
        ),
    ),
)

# for flops calculation
input_shape = [(2, 1, 128), (2, 1, 128), (1, 128, 128)]
