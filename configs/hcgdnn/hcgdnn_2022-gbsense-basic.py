_base_ = [
    '../_base_/default_runtime.py',
]

dataset_type = 'GBSenseBasic'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/GBSense/2022/Basic'
data = dict(
    samples_per_gpu=320,
    workers_per_gpu=40,
    train=dict(
        type=dataset_type,
        file_name='data_1_train.pkl',
        data_root=data_root,
    ),
    val=dict(
        type=dataset_type,
        file_name='data_1_test.pkl',
        data_root=data_root,
    ),
    test=dict(
        type=dataset_type,
        file_name='data_1_test.pkl',
        data_root=data_root,
    ),
)

in_size = 100
out_size = 288
heads = ['CNN', 'BiGRU1', 'BiGRU2']
# Model
model = dict(
    type='DNN',
    method_name='HCGDNN',
    backbone=dict(
        type='HCGNet',
        depth=8,
        heads=heads,
        input_size=in_size,
        avg_pool=(1, 8),
    ),
    classifier_head=dict(
        type='HCGDNNHead',
        in_features=in_size,
        out_features=out_size,
        num_classes=24,
        heads=heads,
        loss_cls=dict(
            type='BinaryCrossEntropyLoss',
            loss_weight=1,
        ),
    ),
)

total_epochs = 800

# Optimizer
optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    gamma=0.3,
    step=[800])
