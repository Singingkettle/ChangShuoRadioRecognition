_base_ = '../_base_/default_runtime.py'

dataset_type = 'WTIRILDataset'
data_root = '/home/raolu/data_200k'
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.json',
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
    ),
)

# Model
model = dict(
    type='RILCLDNN',
    backbone=dict(
        type='CRNet',
        in_channels=1,
        in_height=8,
        avg_pool=(1, 8),
        cnn_depth=4,
        rnn_depth=2,
        input_size=80,
        out_indices=(4,),
        rnn_mode='LSTM',
    ),
    classifier_head=dict(
        type='LocationHead',
        heads=[
            # PX Head
            dict(
                type='AMCHead',
                num_classes=9,
                in_features=50,
                out_features=256,
                loss_cls=dict(
                    type='CustomCrossEntropyLoss',
                    loss_weight=1.0,
                ),
            ),
            # PY Head
            dict(
                type='AMCHead',
                num_classes=9,
                in_features=50,
                out_features=256,
                loss_cls=dict(
                    type='CustomCrossEntropyLoss',
                    loss_weight=1.0,
                ),
            ),
        ]
    ),
)

train_cfg = dict()
test_cfg = dict()

total_epochs = 400
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 12
