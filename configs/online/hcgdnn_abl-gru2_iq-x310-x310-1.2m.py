_base_ = '../_base_/default_runtime.py'

dataset_type = 'WTIMCOnlineDataset'
data_root = '/home/raolu/Data/SignalProcessing/ModulationClassification/Online/ModulationClassification_x310_x310_1.2m'
data = dict(
    samples_per_gpu=640,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.json',
        channel_mode=True,
        use_cache=True,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val.json',
        channel_mode=True,
        use_cache=True,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val.json',
        channel_mode=True,
        use_cache=True,
    ),
)

in_size = 100
out_size = 288
# Model
model = dict(
    type='HCGDNN',
    backbone=dict(
        type='HCGNetGRU2',
        input_size=in_size,
    ),
    classifier_head=dict(
        type='AMCHead',
        in_features=in_size,
        out_features=out_size,
        num_classes=8,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1,
        ),
    ),
)

train_cfg = dict()
test_cfg = dict()

total_epochs = 1600

# Optimizer
optimizer = dict(type='Adam', lr=0.00044)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    gamma=0.3,
    step=[800])
