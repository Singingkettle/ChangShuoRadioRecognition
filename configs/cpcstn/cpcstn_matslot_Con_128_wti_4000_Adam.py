_base_ = '../_base_/default_runtime.py'

# Dataset
window_size = 128
num_slot = 9
dataset_type = 'SlotDatasetV2'
data_root = '/home/zry/Data/SignalProcessing/ModulationClassification/WTISLOT/20210111A'
data = dict(
    samples_per_gpu=10,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        window_size=window_size,
        num_slot=num_slot,
        ann_file='train_and_val.json',
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        window_size=window_size,
        num_slot=num_slot,
        ann_file='test.json',
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        window_size=window_size,
        num_slot=num_slot,
        ann_file='test.json',
    ),
)

# Model
model = dict(
    type='CPCSTN',
    backbone=dict(
        type='CPCSTN',
        num_slot=num_slot,
        num_filter=3,
        slot_size=window_size,
    ),
    classifier_head=dict(
        type='DSAMCHead',
        num_classes=11,
        in_features=16,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
    ),
)

train_cfg = dict()
test_cfg = dict()

total_epochs = 200
evaluation = dict(interval=10)
# Optimizer
#optimizer = dict(type='RMSprop', lr=0.001, alpha=0.9, eps=1e-07)
optimizer = dict(type='Adam')
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')
