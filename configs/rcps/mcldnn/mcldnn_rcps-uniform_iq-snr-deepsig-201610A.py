_base_ = [
    '../_base_/models/mcldnn_iq-snr-deepsig-201610A.py',
]

work_dir = '/home/citybuster/Data/RCPS/work_dirs/amc/deepsig201610A/mcldnn_rcps-uniform'

method_name = 'rcps_uniform_power'

model = dict(
    head=dict(
        loss=dict(
            type='RCPSCrossEntropyLoss',
            reliability_key='snr',
            reliability_map=dict(type='linear', min=-20, max=18),
            epsilon=dict(type='power', max=1.0, gamma=1.0),
            base=dict(type='uniform'),
            sample_weight=dict(type='none'),
            loss_weight=1.0),
    ),
)
