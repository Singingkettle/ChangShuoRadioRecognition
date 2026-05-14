_base_ = [
    '../petcgdnn/petcgdnn_rcps-entropy_iq-snr-deepsig-201610A.py',
]

work_dir = '/home/citybuster/Data/RCPS/work_dirs/parity/amc/deepsig201610A/petcgdnn_rcps-entropy_kerasinit'
method_name = 'rcps_entropy_kerasinit'

model = dict(
    backbone=dict(
        init_cfg=[
            dict(type='Xavier', layer='Conv2d', distribution='uniform'),
            dict(type='Xavier', layer='Linear', distribution='uniform'),
            dict(type='RNN', layer='GRU', gain=1.0),
        ],
    ),
    head=dict(
        loss=dict(
            epsilon=dict(
                source='/home/citybuster/Data/RCPS/work_dirs/rcps_tables/deepsig201610A/petcgdnn_kerasinit_seed2026_entropy_match.npz'))),
)
