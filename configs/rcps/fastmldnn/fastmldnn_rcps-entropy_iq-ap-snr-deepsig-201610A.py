_base_ = ['../_base_/models/fastmldnn_iq-ap-snr-deepsig-201610A.py']

work_dir = '/home/citybuster/Data/RCPS/work_dirs/amc/deepsig201610A/fastmldnn_rcps-entropy'
method_name = 'rcps_entropy_match'

model = dict(
    head=dict(
        loss=dict(
            type='RCPSCrossEntropyLoss',
            reliability_key='snr',
            reliability_map=dict(type='linear', min=-20, max=18),
            epsilon=dict(
                type='entropy_match',
                source='/home/citybuster/Data/RCPS/work_dirs/rcps_tables/deepsig201610A/fastmldnn_hard-ce_seed2026_entropy_match.npz'),
            base=dict(type='uniform'),
            sample_weight=dict(type='none'),
            loss_weight=1.0)))
