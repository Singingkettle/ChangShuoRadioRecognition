_base_ = '../_base_/default_runtime.py'

dataset_type = 'WTISEIDataset'
data_root = '/home/citybuster/Data/SignalProcessing/SpecificEmitterIdentification/Chuan2021.04.22'
data = dict(
    samples_per_gpu=80,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        modulation_list=['64qam'],
        ann_file='train_and_val.json',
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        modulation_list=['64qam'],
        ann_file='test.json',
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        modulation_list=['64qam'],
        ann_file='test.json',
    ),
)

model = dict(
    type='MBRFI',
    backbone=dict(
        type='AlexNetCO',
    ),
    classifier_head=dict(
        type='SEIHead',
        num_classes=5,
        in_features=256,
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
optimizer = dict(type='Adam', lr=0.00005)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')
