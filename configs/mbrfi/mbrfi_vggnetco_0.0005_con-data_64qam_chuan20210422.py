_base_ = '../_base_/default_runtime.py'

dataset_type = 'WTISEIDataset'
data_root = '/home/citybuster/Data/SignalProcessing/SpecificEmitterIdentification/Chuan2021.04.22'
data = dict(
    samples_per_gpu=10,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        modulation_list=['64qam'],
        ann_file='train_and_val.json',
        is_dual=True,
        return_dual_label=False,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        modulation_list=['64qam'],
        ann_file='test.json',
        is_dual=True,
        return_dual_label=False,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        modulation_list=['64qam'],
        ann_file='test.json',
        is_dual=True,
        return_dual_label=False,
    ),
)

model = dict(
    type='MBRFI',
    is_dual=True,
    return_dual_label=False,
    backbone=dict(
        type='VGGNetCO',
    ),
    classifier_head=dict(
        type='SEIHead',
        num_classes=5,
        in_features=512,
        out_features=256,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1,
        ),
    ),
)
train_cfg = dict()
test_cfg = dict()

total_epochs = 600
# optimizer
optimizer = dict(type='Adam', lr=0.00005)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[400, ])
