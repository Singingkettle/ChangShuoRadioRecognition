log_level = 'INFO'
dataset_type = 'FBDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201610A'
data = dict(
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train_and_val.json',
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
    ),
)

model = dict(
    type='Tree',
    max_depth=6,
)
