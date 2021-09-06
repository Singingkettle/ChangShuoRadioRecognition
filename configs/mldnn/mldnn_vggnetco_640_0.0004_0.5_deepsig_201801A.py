_base_ = '../_base_/default_runtime.py'

dataset_type = 'WTIMCDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201801A'
data = dict(
    samples_per_gpu=320,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train_and_val.json',
        iq=False,
        ap=False,
        co=True,
        filter_config=1,
        use_compress=True,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        iq=False,
        ap=False,
        co=True,
        filter_config=1,
        use_compress=True,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        iq=False,
        ap=False,
        co=True,
        filter_config=1,
    ),
)

model = dict(
    type='CNNCO',
    backbone=dict(
        type='VGGNetCO',
    ),
    classifier_head=dict(
        type='AMCHead',
        num_classes=24,
        in_features=512,
        out_features=256,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1,
        ),
    ),
)
train_cfg = dict()
test_cfg = dict()

total_epochs = 400
# optimizer
optimizer = dict(type='Adam', lr=0.00002)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')
