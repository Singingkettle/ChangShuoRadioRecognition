_base_ = '../_base_/default_runtime.py'

dataset_type = 'WTIMCDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201610A'
data = dict(
    samples_per_gpu=640,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train_and_val.json',
        channel_mode=True,
        merge_res=True,
        use_cache=True,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        channel_mode=True,
        merge_res=True,
        use_cache=True,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        channel_mode=True,
        merge_res=True,
        use_cache=True,
    ),
)

in_size = 40
out_size = 256
# Model
model = dict(
    type='MCT',
    backbone=dict(
        type='MCTNetV2',
        ninp=in_size,
    ),
    classifier_head=dict(
        type='AMCHead',
        in_features=in_size,
        out_features=out_size,
        num_classes=11,
        loss_cls=dict(
            type='FocalLoss',
            loss_weight=1,
            alpha=0.5,
        ),
    ),
)

train_cfg = dict()
test_cfg = dict()

total_epochs = 1600

# Optimizer
optimizer = dict(type='Adam', lr=0.0004)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')
