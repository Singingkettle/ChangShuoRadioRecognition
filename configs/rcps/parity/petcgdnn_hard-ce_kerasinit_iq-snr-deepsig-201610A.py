_base_ = [
    '../petcgdnn/petcgdnn_hard-ce_iq-snr-deepsig-201610A.py',
]

work_dir = '/home/citybuster/Data/RCPS/work_dirs/parity/amc/deepsig201610A/petcgdnn_hard-ce_kerasinit'
method_name = 'hard_ce_kerasinit'

# Parity-only variant: Keras PET-CGDNN uses glorot_uniform Conv/Dense
# initializers and CuDNNGRU defaults. This config isolates initialization
# without changing architecture, data split, optimizer, or loss.
model = dict(
    backbone=dict(
        init_cfg=[
            dict(type='Xavier', layer='Conv2d', distribution='uniform'),
            dict(type='Xavier', layer='Linear', distribution='uniform'),
            dict(type='RNN', layer='GRU', gain=1.0),
        ],
    ),
)
