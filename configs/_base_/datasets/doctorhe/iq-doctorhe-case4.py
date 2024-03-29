dataset_type = 'DoctorHeDataset'
data_root = './data/ModulationClassification/DoctorHe/2023.03.A'
case = 'case4'
data_info = 'time_domain'
data = dict(
    samples_per_gpu=640,
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
