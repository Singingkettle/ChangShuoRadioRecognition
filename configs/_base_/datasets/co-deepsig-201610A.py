dataset_type = 'DeepSigDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201610A'
data = dict(
    samples_per_gpu=320,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file='train_and_validation.json',
        pipeline=[
            dict(type='LoadConstellationFromCache', data_root=data_root,
                 filename='train_and_validation_filter_size_0.020_stride_0.020.pkl', to_float32=True),
            dict(type='LoadAnnotations'),
            dict(type='Collect', keys=['cos', 'mod_labels'])
        ],
        data_root=data_root,
    ),
    val=dict(
        type=dataset_type,
        ann_file='test.json',
        pipeline=[
            dict(type='LoadConstellationFromCache', data_root=data_root,
                 filename='test_filter_size_0.020_stride_0.020.pkl', to_float32=True),
            dict(type='Collect', keys=['cos'])
        ],
        data_root=data_root,
        evaluate=[
            dict(type='EvaluateModulationPrediction', )
        ],
    ),
    test=dict(
        type=dataset_type,
        ann_file='test.json',
        pipeline=[
            dict(type='LoadConstellationFromCache', data_root=data_root,
                 filename='test_filter_size_0.020_stride_0.020.pkl', to_float32=True),
            dict(type='Collect', keys=['cos'])
        ],
        data_root=data_root,
        evaluate=[
            dict(type='EvaluateModulationPrediction', )
        ],
        save=[
            dict(type='SaveModulationPrediction', )
        ],
    ),
)
