_base_ = [
    '../_base_/datasets/slot_co_data.py',
    '../_base_/default_runtime.py']

model = dict(
    type='CNNCO',
    backbone=dict(
        type='GoogleNetCO',
    ),
    classifier_head=dict(
        type='AMCHead',
        num_classes=8,
        in_features=1024,
        out_features=256,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1,
        ),
    ),
)
train_cfg = dict()
test_cfg = dict()

total_epochs = 400
# optimizer
optimizer = dict(type='Adam', lr=0.00001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')
