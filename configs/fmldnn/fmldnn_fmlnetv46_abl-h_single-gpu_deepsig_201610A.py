_base_ = '../_base_/default_runtime.py'

batch_size = 640
channel_mode = False
use_cache = True
dataset_type = 'WTIMCDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201610A'
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train_and_val.json',
        iq=True,
        ap=True,
        channel_mode=channel_mode,
        use_cache=use_cache,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        iq=True,
        ap=True,
        channel_mode=channel_mode,
        use_cache=use_cache,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        iq=True,
        ap=True,
        channel_mode=channel_mode,
    ),
)

in_features = 100
out_features = 256
num_classes = 11
model = dict(
    type='FMLDNN',
    channel_mode=channel_mode,
    backbone=dict(
        type='FMLNetV46',
        in_features=4,
        channel_mode=channel_mode,
        skip_connection=False,
    ),
    classifier_head=dict(
        type='AMCHead',
        in_features=in_features,
        out_features=out_features,
        num_classes=num_classes,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1,
        ),
    ),
)
train_cfg = dict()
test_cfg = dict()

total_epochs = 600
# optimizer
optimizer = dict(type='Adam', lr=0.00069)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    gamma=0.3,
    step=[300, 500])

# for flops calculation
input_shape = [(1, 2, 128), (1, 2, 128), (1, 128, 128)]