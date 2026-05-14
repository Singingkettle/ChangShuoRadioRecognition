_base_ = [
    '../petcgdnn/petcgdnn_static-ls_iq-snr-deepsig-201610A.py',
]

work_dir = '/home/citybuster/Data/RCPS/work_dirs/parity/amc/deepsig201610A/petcgdnn_static-ls_kerasinit'
method_name = 'static_ls_kerasinit'

model = dict(
    backbone=dict(
        init_cfg=[
            dict(type='Xavier', layer='Conv2d', distribution='uniform'),
            dict(type='Xavier', layer='Linear', distribution='uniform'),
            dict(type='RNN', layer='GRU', gain=1.0),
        ],
    ),
)
