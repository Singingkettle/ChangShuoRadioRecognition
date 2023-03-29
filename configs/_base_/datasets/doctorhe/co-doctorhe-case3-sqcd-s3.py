dataset_type = 'DoctorHeDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DoctorHe/2023.03.A'
case = 'case3'
data_info = 'Graymat_SQCD_15dB_shift3step.mat'
data = dict(
    samples_per_gpu=80,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        case=case,
        data_info=data_info,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        case=case,
        data_info=data_info,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        case=case,
        data_info=data_info,
    ),
)