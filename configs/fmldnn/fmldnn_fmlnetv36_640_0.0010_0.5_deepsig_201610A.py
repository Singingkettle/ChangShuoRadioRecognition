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
        use_snr_label=True,
        snr_threshold=0.0,
        item_weights=[0.999, 0.001],  # it's set by snr
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        iq=True,
        ap=True,
        channel_mode=True,
        use_snr_label=True,
        snr_threshold=0.0,
        item_weights=[0.999, 0.001],
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        iq=True,
        ap=True,
        channel_mode=True,
        use_snr_label=True,
        snr_threshold=0.0,
        item_weights=[0.999, 0.001],  # it's set by snr
    ),
)

model = dict(
    type='FMLDNN',
    backbone=dict(
        type='FMLNetV36',
    ),
    classifier_head=dict(
        type='FMLAUXHead',
        heads=[
            # Snr Head
            dict(
                type='AMCHead',
                num_classes=2,
                in_features=156,
                out_features=256,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    loss_weight=1,
                ),
            ),
            # Low Head
            dict(
                type='AMCHead',
                num_classes=11,
                in_features=78,
                out_features=256,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    loss_weight=1,
                ),
            ),
            # High Head
            dict(
                type='AMCHead',
                num_classes=11,
                in_features=78,
                out_features=256,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    loss_weight=1,
                ),
            ),
            # Merge Head
            dict(
                type='MergeAMCHead',
                loss_cls=dict(
                    type='NLLLoss',
                    loss_weight=0,
                ),
            ),
            # Inter Head
            dict(
                type='InterOrthogonalHead',
                num_bases=2,
                batch_size=batch_size,
                bmm='cosine',
                is_abs=True,
                loss_aux=dict(
                    type='LogisticLoss',
                    loss_weight=1
                ),
            ),
            # Intra Head
            dict(
                type='IntraOrthogonalHead',
                in_features=156,  # keep the same as snr head
                batch_size=batch_size,  # keep the same as samples_per_gpu
                num_classes=11,
                mm='cosine',
                is_abs=True,
                loss_aux=dict(
                    type='LogisticLoss',
                    loss_weight=1
                ),
            ),
        ]
    ),
)
train_cfg = dict()
test_cfg = dict()

total_epochs = 400
# optimizer
optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')

# find_unused_parameters =True
