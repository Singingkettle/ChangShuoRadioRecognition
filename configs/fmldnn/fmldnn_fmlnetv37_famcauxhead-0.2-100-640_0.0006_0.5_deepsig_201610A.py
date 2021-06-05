_base_ = '../_base_/default_runtime.py'

batch_size = 640
dataset_type = 'WTIMCDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201610A'
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train_and_val.json',
        iq=True,
        ap=True,
        channel_mode=True,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        iq=True,
        ap=True,
        channel_mode=True,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        iq=True,
        ap=True,
        channel_mode=True,
    ),
)

in_features = 1200
out_features = 256
num_classes = 11
model = dict(
    type='FMLDNN',
    backbone=dict(
        type='FMLNetV37',
    ),
    classifier_head=dict(
        type='FAMCAUXHead',
        in_features=in_features,
        out_features=out_features,
        num_classes=num_classes,
        batch_size=batch_size,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1,
        ),
        # Intra Head
        aux_head=dict(
            type='IntraOrthogonalHead',
            in_features=out_features,
            batch_size=batch_size,
            num_classes=num_classes,
            mm='inner_product',
            is_abs=False,
            loss_aux=dict(
                type='LogisticLoss',
                loss_weight=0.2,
                temperature=100,
            ),
        ),
    ),
)
train_cfg = dict()
test_cfg = dict()

total_epochs = 400
# optimizer
optimizer = dict(type='Adam', lr=0.0006)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')

evaluation = dict(interval=1)
