_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/iq-channel-snr-[-8,20]-deepsig-201801A.py',
]

in_size = 100
out_size = 288
heads = ['CNN', 'BiGRU1', 'BiGRU2']
# Model
model = dict(
    type='SingleHeadClassifier',
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


total_epochs = 800

# Optimizer
optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=None)
# learning policy
# learning policy
lr_config = dict(policy='fixed')