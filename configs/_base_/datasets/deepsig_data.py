dataset_type = 'WTIMCDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201610A'
data = dict(
    samples_per_gpu=1024,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.json',
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val.json',
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
    ),
)
evaluation = dict(interval=1)
