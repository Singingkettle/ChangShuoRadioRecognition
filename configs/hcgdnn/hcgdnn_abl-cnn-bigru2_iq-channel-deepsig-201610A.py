_base_ = [
    './schedule.py',
    './data_iq-channel-deepsig-201610A.py',
    '../_base_/default_runtime.py',
]

in_size = 100
out_size = 288
heads = ['CNN', 'BiGRU2']
# Model
model = dict(
    type='SingleHeadClassifier',
    method_name='HCGDNN-V5',
    backbone=dict(
        type='HCGNet',
        heads=heads,
        input_size=in_size,
    ),
    classifier_head=dict(
        type='HCGDNNHead',
        in_features=in_size,
        out_features=out_size,
        num_classes=11,
        heads=heads,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1,
        ),
    ),
)
