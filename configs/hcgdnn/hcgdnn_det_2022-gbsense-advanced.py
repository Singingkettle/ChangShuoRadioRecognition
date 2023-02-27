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
            type='det',

        )
    ),
    val=dict(
        type=dataset_type,
        file_name='data_2_test.h5',
        data_root=data_root,
        head=dict(
            type='det',
        )
    ),
    test=dict(
        type=dataset_type,
        file_name='data_2_test.h5',
        data_root=data_root,
        head=dict(
            type='det',
        )
    ),
)

in_size = 100
out_size = 288
# Model
model = dict(
    type='SingleHeadClassifier',

    backbone=dict(
        type='HCGNetGRU2',
        depth=16,
        input_size=in_size,
        avg_pool=(1, 8),
    ),
    classifier_head=dict(
        type='GBDetHead',
        channel_cls_num=24,
        mod_cls_num=13,
        in_features=in_size,
        out_features=out_size,
        is_share=False,
        loss_channel=dict(
            type='BinaryCrossEntropyLoss',
            loss_weight=1,
        ),
        loss_mod=dict(
            type='BinaryCrossEntropyLoss',
            loss_weight=1,
        ),
    ),
)

total_epochs = 40000

# Optimizer
optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')
