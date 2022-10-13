_base_ = [
    './schedule.py',
    '../_base_/default_runtime.py',
    './data_iq-channel-rms-deepsig-201610A.py',
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
    ),
    classifier_head=dict(
        type='HCGDNNHead',
        in_features=in_size,
        out_features=out_size,
        num_classes=11,
        heads=heads,
        loss_cls=dict(
            type='BinaryCrossEntropyLoss',
            loss_weight=1,
        ),
    ),
)
