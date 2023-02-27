log_level = 'INFO'

dataset_type = 'DeepSigDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201801A'
data = dict(
    samples_per_gpu=80,
    workers_per_gpu=20,
    train=dict(
        type=dataset_type,
        ann_file='train_and_validation.json',
        preprocess=[
            dict(type='FilterBySNR', snr_set=[snr for snr in range(-8, 22, 2)]),
        ],
        pipeline=[
            dict(type='LoadIQFromCache', data_root=data_root, file_name='train_and_validation_iq.pkl', to_float32=True),
            dict(type='Cumulants'),
            dict(type='LoadAnnotations'),
            dict(type='Collect', keys=['cls', 'mod_labels'])
        ],
        data_root=data_root,
    ),
    val=dict(
        type=dataset_type,
        ann_file='test.json',
        preprocess=[
            dict(type='FilterBySNR', snr_set=[snr for snr in range(-8, 22, 2)]),
        ],
        pipeline=[
            dict(type='LoadIQFromCache', data_root=data_root, file_name='test_iq.pkl', to_float32=True),
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
        preprocess=[
            dict(type='FilterBySNR', snr_set=[snr for snr in range(-8, 22, 2)]),
        ],
        pipeline=[
            dict(type='LoadIQFromCache', data_root=data_root, file_name='test_iq.pkl', to_float32=True),
            dict(type='Cumulants'),
            dict(type='Collect', keys=['inputs'])
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
