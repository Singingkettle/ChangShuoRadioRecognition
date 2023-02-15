dataset_type = 'CSSSDetSingleStage'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/CSSS/data'
snr_start = -8
snr_end = 30
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        file_info=dict(
            file_prefixes=['train', 'val'],
            snrs=[i for i in range(snr_start, snr_end, 2)]
        ),
        data_root=data_root,
    ),
    val=dict(
        type=dataset_type,
        file_info=dict(
            file_prefixes=['test'],
            snrs=[i for i in range(snr_start, snr_end, 2)]
        ),
        data_root=data_root,
    ),
    test=dict(
        type=dataset_type,
        file_info=dict(
            file_prefixes=['test'],
            snrs=[i for i in range(snr_start, snr_end, 2)]
        ),
        data_root=data_root,
    ),
)
