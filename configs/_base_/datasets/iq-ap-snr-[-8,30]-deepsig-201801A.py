dataset_type = 'DeepSigDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201801A'
data = dict(
    samples_per_gpu=80,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file='train_and_validation.json',
        augment=[
            dict(type='FilterBySNR', low_snr=-8, high_snr=30),
        ],
        pipeline=[
            dict(type='LoadIQFromHDF5', data_root=data_root, filename='train_and_validation_iq.h5', to_float32=True),
            dict(type='LoadAPFromHDF5', data_root=data_root, filename='train_and_validation_ap.h5', to_float32=True),
            dict(type='LoadAnnotations'),
            dict(type='Collect', keys=['iqs', 'aps', 'mod_labels'])
        ],
        data_root=data_root,
    ),
    val=dict(
        type=dataset_type,
        ann_file='test.json',
        augment=[
            dict(type='FilterBySNR', low_snr=-8, high_snr=30),
        ],
        pipeline=[
            dict(type='LoadIQFromHDF5', data_root=data_root, filename='test_iq.h5', to_float32=True),
            dict(type='LoadAPFromHDF5', data_root=data_root, filename='test_ap.h5', to_float32=True),
            dict(type='Collect', keys=['iqs', 'aps'])
        ],
        data_root=data_root,
        evaluate=[
            dict(type='EvaluateModulationPrediction', )
        ],
    ),
    test=dict(
        type=dataset_type,
        ann_file='test.json',
        augment=[
            dict(type='FilterBySNR', low_snr=-8, high_snr=30),
        ],
        pipeline=[
            dict(type='LoadIQFromHDF5', data_root=data_root, filename='test_iq.h5', to_float32=True),
            dict(type='LoadAPFromHDF5', data_root=data_root, filename='test_ap.h5', to_float32=True),
            dict(type='Collect', keys=['iqs', 'aps'])
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
