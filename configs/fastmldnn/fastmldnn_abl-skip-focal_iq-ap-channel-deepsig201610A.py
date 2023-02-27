_base_ = [
    './schedule.py',
    '../_base_/default_runtime.py',
    './data_iq-ap-channel-deepsig201610A.py'
]

in_features = 100
out_features = 256
num_classes = 11
model = dict(
    type='SingleHeadClassifier',

    backbone=dict(
        type='FMLNet',
        in_features=4,
        channel_mode=True,
        skip_connection=True,
    ),
    classifier_head=dict(
        type='ACMHead',
        in_features=in_features,
        out_features=out_features,
        num_classes=num_classes,
        loss_cls=dict(
            type='FocalLoss',
            loss_weight=1,
            alpha=0.5,
        ),
    ),
)
train_cfg = dict()
test_cfg = dict(vis_fea=True)

# for flops calculation
input_shape = [(2, 1, 128), (2, 1, 128), (1, 128, 128)]
