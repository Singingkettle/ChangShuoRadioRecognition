_base_ = [
    '../_base_/default_runtime.py',
]

dataset_type = 'GBSenseAdvancedChannel'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/GBSense/2022/Advanced'
data = dict(
    samples_per_gpu=640,
    workers_per_gpu=10,
    train=dict(
        type=dataset_type,
        file_name='train_channel_label.pkl',
        data_root=data_root,
    ),
    val=dict(
        type=dataset_type,
        file_name='test_channel_label.pkl',
        data_root=data_root,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
    ),
)

in_size = 100
out_size = 288
# Model
model = dict(
    type='HCGDNN',
    backbone=dict(
        type='HCGNetGRU2',
        depth=16,
        input_size=in_size,
        avg_pool=(1, 8),
    ),
    classifier_head=dict(
        type='GBBCEHead',
        in_size=in_size,
        out_size=out_size,
        num_classes=1,
        loss_cls=dict(
            type='BinaryCrossEntropyLoss',
            loss_weight=1,
        ),
    ),
)

total_epochs = 400

# Optimizer
optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')
