log_level = 'INFO'
dataset_type = 'FBSlotDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/MATSLOT/IQ_128_Ds10_36000'
data = dict(
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.json',
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
