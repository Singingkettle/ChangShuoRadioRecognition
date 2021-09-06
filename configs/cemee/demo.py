_base_ = '../_base_/default_runtime.py'

batch_size = 256
# Dataset
dataset_type = 'SEICEMEEDataset'
data_root = '/home/citybuster/Data/SignalProcessing/SpecificEmitterIdentification/CEMEE/2021-07'
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        mat_file='1_train.mat',
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        mat_file='1_test.mat',
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        mat_file='Task_1_Test.mat',
    ),
)

in_features = 100
out_features = 256
num_classes = 100
# Model
model = dict(
    type='CEMEE',
    backbone=dict(
        type='CEMEENet',
    ),
    classifier_head=dict(
        type='FAMCAUXHead',
        in_features=in_features,
        out_features=out_features,
        num_classes=num_classes,
        batch_size=batch_size,
        loss_cls=dict(
            type='FocalLoss',
            loss_weight=1,
            alpha=0.5,
        ),
        # Intra Head
        aux_head=dict(
            type='IntraOrthogonalHead',
            in_features=out_features,
            batch_size=batch_size,
            num_classes=num_classes,
            mm='cosine',
            is_abs=False,
            loss_aux=dict(
                type='LogisticLoss',
                loss_weight=0.4,
                temperature=0.7,
            ),
        ),
    ),
)

train_cfg = dict()
test_cfg = dict()

total_epochs = 400

# Optimizer
optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')
