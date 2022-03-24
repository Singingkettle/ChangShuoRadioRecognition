_base_ = [
    '../_base_/datasets/iq-deepsig-201801A.py',
    '../_base_/default_runtime.py'
]

# Model
model = dict(
    type='DNN',
    method_name='CNN2',
    backbone=dict(
        type='CNNNet',
        depth=2,
        in_channels=1,
        out_indices=(1,),
        avg_pool=(1, 8),
    ),
    classifier_head=dict(
        type='AMCHead',
        num_classes=24,
        in_features=10240,
        out_features=256,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
    ),
)

train_cfg = dict()
test_cfg = dict()

total_epochs = 400

# Optimizer
optimizer = dict(type='Adam')
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')
