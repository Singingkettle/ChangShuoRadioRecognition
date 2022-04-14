_base_ = [
    './schedule.py',
    './data_iq-channel-snr-[-8,30]-deepsig-201610A.py',
    '../_base_/default_runtime.py',
]

in_size = 100
out_size = 288
heads = ['CNN', 'BiGRU1', 'BiGRU2']
# Model
model = dict(
    type='DNN',
    method_name='HCGDNN',
    backbone=dict(
        type='HCGNet',
        heads=heads,
        input_size=in_size,
        avg_pool=(1, 8),
    ),
    classifier_head=dict(
        type='HCGDNNHead',
        in_features=in_size,
        out_features=out_size,
        num_classes=24,
        heads=heads,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1,
        ),
    ),
)
