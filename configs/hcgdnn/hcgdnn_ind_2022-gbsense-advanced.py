_base_ = [
    '../_base_/default_runtime.py',
]

dataset_type = 'GBSenseAdvanced'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/GBSense/2022/Advanced'
data = dict(
    samples_per_gpu=640,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        file_name='data_2_train.h5',
        data_root=data_root,
        head=dict(
            type='ind',
        )
    ),
    val=dict(
        type=dataset_type,
        file_name='data_2_test.h5',
        data_root=data_root,
        head=dict(
            type='ind',
        )
    ),
    test=dict(
        type=dataset_type,
        file_name='data_2_test.h5',
        data_root=data_root,
        head=dict(
            type='ind',
        )
    ),
)

in_size = 100
out_size = 288
# Model
model = dict(
    type='DNN',
    method_name='HCGDNN',
    backbone=dict(
        type='HCGNetGRU2',
        depth=16,
        input_size=in_size,
        avg_pool=(1, 8),
    ),
    classifier_head=dict(
        type='GBIndHead',
        channel_head=dict(
            type='AMCHead',
            in_features=in_size,
            out_features=out_size,
            num_classes=24,
            loss_cls=dict(
                type='BinaryCrossEntropyLoss',
                loss_weight=0.1,
            ),
        ),
        mod_head=dict(
            type='AMCHead',
            in_features=in_size,
            out_features=out_size,
            num_classes=13,
            loss_cls=dict(
                type='BinaryCrossEntropyLoss',
                loss_weight=0.1,
            ),
        ),
        order_head=dict(
            type='AMCHead',
            in_features=in_size,
            out_features=out_size,
            num_classes=4,
            loss_cls=dict(
                type='FocalLoss',
                loss_weight=10,
            ),
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
    gamma=0.3,
    step=[300, 600])
