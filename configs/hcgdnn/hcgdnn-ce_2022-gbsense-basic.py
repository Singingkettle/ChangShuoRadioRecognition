_base_ = [
    '../_base_/default_runtime.py',
]

dataset_type = 'GBSenseBasic'
data_root = './data/ModulationClassification/GBSense/2022/Basic'
data = dict(
    samples_per_gpu=640,
    workers_per_gpu=20,
    train=dict(
        type=dataset_type,
        file_name='data_1_train.h5',
        data_root=data_root,
    ),
    val=dict(
        type=dataset_type,
        file_name='data_1_test.h5',
        data_root=data_root,
    ),
    test=dict(
        type=dataset_type,
        file_name='data_1_test.h5',
        data_root=data_root,
    ),
)

in_size = 100
out_size = 288
heads = ['CNN', 'BiGRU1', 'BiGRU2']
# Model
model = dict(
    type='HCGDNN',
    backbone=dict(
        type='HCGNet',
        depth=16,
        heads=heads,
        input_size=in_size,
        has_stride=True,
    ),
    classifier_head=dict(
        type='HCGDNNHead',
        in_size=in_size,
        out_size=out_size,
        num_classes=13,
        heads=heads,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=10,
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
    step=[300, 500])
