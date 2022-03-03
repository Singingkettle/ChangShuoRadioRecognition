_base_ = [
    '../_base_/datasets/slot_iq_data.py',
    '../_base_/default_runtime.py']

model = dict(
    type='DensCNN',
    backbone=dict(
        type='DensNet',
    ),
    classifier_head=dict(
        type='AMCHead',
        num_classes=8,
        in_features=10240,
        out_features=128,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
    ),
)
train_cfg = dict()
test_cfg = dict()

total_epochs = 400
# optimizer
optimizer = dict(type='Adam')
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')