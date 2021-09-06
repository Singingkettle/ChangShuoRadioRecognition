_base_ = '../_base_/default_runtime.py'

batch_size = 640
channel_mode = True
dataset_type = 'WTIMCDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201610A'
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=20,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train_and_val.json',
        iq=True,
        ap=True,
        channel_mode=channel_mode,
        use_cache=True,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        iq=True,
        ap=True,
        channel_mode=channel_mode,
        use_cache=True,
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
        skip_connection=True,
    ),
    classifier_head=dict(
        type='AMCHead',
        in_features=in_features,
        out_features=out_features,
        num_classes=num_classes,
        loss_cls=dict(
            type='FocalLoss',
            loss_weight=1,
            alpha=0.5,
        ),
    ),
)
train_cfg = dict()
test_cfg = dict(vis_fea=True)

total_epochs = 600
# optimizer
optimizer = dict(type='Adam', lr=0.00069)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    gamma=0.3,
    step=[300, 500])
