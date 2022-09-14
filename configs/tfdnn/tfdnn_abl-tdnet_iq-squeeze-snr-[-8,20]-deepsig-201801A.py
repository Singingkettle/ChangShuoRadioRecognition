_base_ = [
    '../_base_/default_runtime.py',
    './iq-squeeze-snr-[-8,20]-deepsig-201801A.py',
]

in_size = 80
out_size = 288
# Model
model = dict(
    type='DNN',
    method_name='TFDNN',
    backbone=dict(
        type='TDNet',
        input_size=in_size,
    ),
    classifier_head=dict(
        type='AMCHead',
        in_features=in_size,
        out_features=out_size,
        num_classes=24,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1,
        ),
    ),
)


total_epochs = 800

# Optimizer
optimizer = dict(type='Adam', lr=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')