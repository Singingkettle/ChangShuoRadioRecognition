dataset_type = 'DeepSigDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201610A'
data = dict(
    samples_per_gpu=320,
    workers_per_gpu=4, persistent_workers=True, prefetch_factor=3,
    train=dict(
        type=dataset_type,
        ann_file='train_and_validation.json',

        pipeline=[
            dict(type='LoadIQFromCache', data_root=data_root, filename='train_and_validation_iq.pkl', to_float32=True),
            dict(type='LoadAPFromCache', data_root=data_root, filename='train_and_validation_ap.pkl', to_float32=True,
                 to_norm=True),
            dict(type='LoadAnnotations'),
            dict(type='Collect', keys=['iqs', 'aps', 'mod_labels'])
        ],
        data_root=data_root,
    ),
    val=dict(
        type=dataset_type,
        ann_file='test.json',
        pipeline=[
            dict(type='LoadIQFromCache', data_root=data_root, filename='test_iq.pkl', to_float32=True),
            dict(type='LoadAPFromCache', data_root=data_root, filename='test_ap.pkl', to_float32=True),
            dict(type='Collect', keys=['iqs', 'aps'])
        ],
        data_root=data_root,
        evaluate=[
            dict(type='EvaluateClassificationWithSNR', )
        ],
    ),
    test=dict(
        type=dataset_type,
        ann_file='test.json',

        pipeline=[
            dict(type='LoadIQFromCache', data_root=data_root, filename='test_iq.pkl', to_float32=True),
            dict(type='LoadAPFromCache', data_root=data_root, filename='test_ap.pkl', to_float32=True),
            dict(type='Collect', keys=['iqs', 'aps'])
        ],
        data_root=data_root,
        evaluate=[
            dict(type='EvaluateClassificationWithSNR', )
        ],
        save=[
            dict(type='SaveModulationPrediction', )
        ],
    ),
)
