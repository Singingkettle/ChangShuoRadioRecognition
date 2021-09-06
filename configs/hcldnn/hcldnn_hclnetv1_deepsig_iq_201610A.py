_base_ = '../_base_/default_runtime.py'

dataset_type = 'WTIMCDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201610A'
data = dict(
    samples_per_gpu=80,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train_and_val.json',
        use_cache=True,
        channel_mode=True,
        merge_res=True,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        use_cache=True,
        channel_mode=True,
        merge_res=True,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        use_cache=True,
        channel_mode=True,
        merge_res=True,
    ),
)

# Model
model = dict(
    type='HCLDNN',
    backbone=dict(
        type='HCLNetV1',
    ),
    classifier_head=dict(
        type='HAMCHead',
        in_features=80,
        out_features=256,
        num_classes=11,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1,
        ),
    ),
)

train_cfg = dict()
test_cfg = dict()

total_epochs = 1000

# Optimizer
optimizer = dict(type='Adam', lr=0.0005)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')
