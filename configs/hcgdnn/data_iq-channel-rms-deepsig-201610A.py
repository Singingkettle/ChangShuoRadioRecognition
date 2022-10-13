dataset_type = 'DeepSigDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201610A'
data = dict(
    samples_per_gpu=320,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file='train_and_validation.json',
        pipeline=[
            dict(type='LoadIQFromCache', data_root=data_root, filename='train_and_validation_iq.pkl', to_float32=True),
            dict(type='ChannelMode', ),
            dict(type='SigmoidLossWeight', alpha=0.2),
            dict(type='RebaseModLabelBySNR', alpha=0.4, beta=20),
            dict(type='Collect', keys=['iqs', 'mod_labels']),
        ],
        data_root=data_root,
    ),
    val=dict(
        type=dataset_type,
        ann_file='test.json',
        pipeline=[
            dict(type='LoadIQFromCache', data_root=data_root, filename='test_iq.pkl', to_float32=True),
            dict(type='ChannelMode', ),
            dict(type='Collect', keys=['iqs'])
        ],
        data_root=data_root,
        evaluate=[
            dict(type='EvaluateClassificationWithSNROfHCGDNN', merge=dict(type='Optimization'))
        ],
    ),
    test=dict(
        type=dataset_type,
        ann_file='test.json',
        pipeline=[
            dict(type='LoadIQFromCache', data_root=data_root, filename='test_iq.pkl', to_float32=True),
            dict(type='ChannelMode', ),
            dict(type='Collect', keys=['iqs'])
        ],
        data_root=data_root,
        evaluate=[
            dict(type='EvaluateClassificationWithSNROfHCGDNN', merge=dict(type='Optimization'))
        ],
        save=[
            dict(type='SaveModulationPredictionOfHCGDNN', merge=dict(type='Optimization'))
        ],
    ),
)
