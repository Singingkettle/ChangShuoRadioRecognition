_base_ = [
    '../_base_/default_runtime.py',
    './data_iq-channel-deepsig-201610A.py',
]

in_size = 80
out_size = 288
# Model
model = dict(
    type='DNN',
    method_name='MCT',
    backbone=dict(
        type='MCTNet',
        input_size=in_size,
    ),
    classifier_head=dict(
        type='AMCHead',
        in_features=in_size,
        out_features=out_size,
        num_classes=11,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1,
        ),
    ),
)


total_epochs = 1600

# Optimizer
optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    gamma=0.1,
    step=[400])