dataset_type = 'DeepSigDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201801A'
data = dict(
    samples_per_gpu=1280,
    workers_per_gpu=20,persistent_workers=True,
    train=dict(
        type=dataset_type,
        ann_file='train_and_validation.json',
        augment=[
            dict(type='FilterBySNR', snr_set=[snr for snr in range(-8, 22, 2)]),
        ],
        pipeline=[
            dict(type='LoadIQFromCache', data_root=data_root, filename='train_and_validation_iq.pkl', to_float32=True),
            dict(type='LoadFTFromIQ', to_float32=True, is_squeeze=True),
            dict(type='LoadAnnotations'),
            dict(type='Collect', keys=['fts', 'mod_labels'])
        ],
        data_root=data_root,
    ),
    val=dict(
        type=dataset_type,
        ann_file='test.json',
        augment=[
            dict(type='FilterBySNR', snr_set=[snr for snr in range(-8, 22, 2)]),
        ],
        pipeline=[
            dict(type='LoadIQFromCache', data_root=data_root, filename='test_iq.pkl', to_float32=True),
            dict(type='LoadFTFromIQ', to_float32=True, is_squeeze=True),
            dict(type='Collect', keys=['fts'])
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
            dict(type='FilterBySNR', snr_set=[snr for snr in range(-8, 22, 2)]),
        ],
        pipeline=[
            dict(type='LoadIQFromCache', data_root=data_root, filename='test_iq.pkl', to_float32=True),
            dict(type='LoadFTFromIQ', to_float32=True, is_squeeze=True),
            dict(type='Collect', keys=['fts'])
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
