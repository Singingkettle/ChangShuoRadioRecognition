_base_ = [
    '../_base_/default_runtime.py',
]
# 0.19, 0.18, 0.63
dataset_type = 'GBSenseAdvanced'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/GBSense/2022/Advanced'
data = dict(
    samples_per_gpu=640,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        file_name=['data_2_train.h5', 'data_2_test.h5'],
        data_root=data_root,
    ),
    val=dict(
        type=dataset_type,
        file_name='data_2_test.h5',
        data_root=data_root,
    ),
    test=dict(
        type=dataset_type,
        file_name='data_2_test.h5',
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
        num_classes=24*13,
        heads=heads,
        loss_cls=dict(
            type='BinaryCrossEntropyLoss',
            loss_weight=10,
        ),
    ),
)

total_epochs = 2000

# Optimizer
optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    gamma=0.3,
    step=[800])
