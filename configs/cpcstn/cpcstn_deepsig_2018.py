_base_ = '../_base_/default_runtime.py'

# Dataset
dataset_type = 'SlotDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201801A'
data = dict(
    samples_per_gpu=120,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train_and_val.json',
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
    ),
)

# Model
model = dict(
    type='CPCSTN',
    backbone=dict(
        type='CPCSTN',
        num_slot=8,
        num_filter=3,
        slot_size=128,
        stn_net_name='v5',
    ),
    classifier_head=dict(
        type='DSAMCHead',
        num_classes=24,
        in_features=16,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
    ),
)

train_cfg = dict()
test_cfg = dict()

total_epochs = 100
evaluation = dict(interval=10)
# Optimizer
optimizer = dict(type='Adam', lr=0.0002)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')
