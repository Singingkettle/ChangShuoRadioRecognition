log_level = 'INFO'

dataset_type = 'DeepSigDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201801A'
data = dict(
    samples_per_gpu=640,
    workers_per_gpu=4, persistent_workers=True, prefetch_factor=3,
    train=dict(
        type=dataset_type,
        ann_file='train_and_validation.json',
        pipeline=[
            dict(type='LoadIQFromCache', data_root=data_root, filename='train_and_validation_iq.pkl', to_float32=True),
            dict(type='Cumulants'),
            dict(type='LoadAnnotations'),
            dict(type='Collect', keys=['cls', 'mod_labels'])
        ],
        data_root=data_root,
    ),
    val=dict(
        type=dataset_type,
        ann_file='test.json',
        pipeline=[
            dict(type='LoadIQFromCache', data_root=data_root, filename='test_iq.pkl', to_float32=True),
            dict(type='Cumulants'),
            dict(type='Collect', keys=['cls'])
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
            dict(type='Cumulants'),
            dict(type='Collect', keys=['iqs'])
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

x = 'cls'
y = 'mod_labels'

model = dict(
    type='SVM',
    regularization=1,
    max_iter=50000,
    method_name='SVM',
)
