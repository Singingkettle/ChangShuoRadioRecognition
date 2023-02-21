_base_ = [
    './schedule.py',
    '../_base_/default_runtime.py',
    '../_base_/datasets/iq-channel-deepsig-201610A.py',
]

in_size = 100
out_size = 288
# Model
model = dict(
    type='SingleHeadClassifier',
    method_name='HCGDNN-V2',
    backbone=dict(
        type='HCGNetGRU1',
        input_size=in_size,
    ),
    classifier_head=dict(
        type='ACMHead',
        in_features=in_size,
        out_features=out_size,
        num_classes=11,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1,
        ),
    ),
)
