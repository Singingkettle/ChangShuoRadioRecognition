_base_ = [
    '../_base_/default_runtime.py',
]
# 0.01, 0.45, 0.54
dataset_type = 'GBSenseBasic'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/GBSense/2022/Basic'
data = dict(
    samples_per_gpu=640,
    workers_per_gpu=20,
    train=dict(
        type=dataset_type,
        file_name=['data_1_train.h5', 'data_1_test.h5'],
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
    type='SingleHeadClassifier',

    backbone=dict(
        type='HCGNet',
        depth=16,
        heads=heads,
        input_size=in_size,
        has_stride=True,
    ),
    classifier_head=dict(
        type='HCGDNNHead',
        in_features=in_size,
        out_features=out_size,
        num_classes=13,
        heads=heads,
        loss_cls=dict(
            type='FocalLoss',
            loss_weight=10,
            alpha=0.5
        ),
    ),
)

total_epochs = 1600

# Optimizer
optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    gamma=0.1,
    step=[300, 500, 800, 1200])
