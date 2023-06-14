dataset_type = 'DeepSigDataset'
data_root = './data/ModulationClassification/DeepSig/201801A'
target_name = 'modulation'
data = dict(
    samples_per_gpu=80,
    workers_per_gpu=10,
    train=dict(
        type=dataset_type,
        ann_file='train_and_validation.json',
        pipeline=[
            dict(type='LoadIQFromFile', to_float32=True),
            dict(type='LoadAPFromIQ'),
            dict(type='LoadAnnotations', target_info={target_name: 'int64'}),
            dict(type='MLDNNSNRLabel'),
            dict(type='Collect', keys=['inputs', 'targets'])
        ],
        data_root=data_root,
    ),
    val=dict(
        type=dataset_type,
        ann_file='test.json',
        pipeline=[
            dict(type='LoadIQFromFile', to_float32=True),
            dict(type='LoadAPFromIQ'),
            dict(type='Collect', keys=['inputs'])
        ],
        data_root=data_root,
        evaluate=[
            dict(type='EvaluateMLDNN'),
        ],
    ),
    test=dict(
        type=dataset_type,
        ann_file='test.json',
        pipeline=[
            dict(type='LoadIQFromFile', to_float32=True),
            dict(type='LoadAPFromIQ'),
            dict(type='Collect', keys=['inputs'])
        ],
        data_root=data_root,
        evaluate=[
            dict(type='EvaluateMLDNN'),
        ],
        format=[
            dict(type='SaveMLDNN'),
        ],
    ),
)
