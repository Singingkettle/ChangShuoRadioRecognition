dataset_type = 'SlotDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/MATSLOT/IQ_128_Ds10_36000'
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.json',
        is_seq=True,
        use_ap=True,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        is_seq=True,
        use_ap=True,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        is_seq=True,
        use_ap=True,
    ),
)