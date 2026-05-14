_base_ = [
    '../mcldnn/mcldnn_hard-ce_iq-snr-deepsig-201610A.py',
]

work_dir = '/home/citybuster/Data/RCPS/work_dirs/parity/amc/deepsig201610A/mcldnn_hard-ce_kerasinit'
method_name = 'hard_ce_kerasinit'

# Parity-only variant: Keras Conv/Dense defaults use glorot_uniform and
# Keras recurrent layers use Glorot input kernels with orthogonal recurrent
# kernels. This config tests whether initialization explains the gap between
# AMR-Benchmark and the PyTorch reproduction.
model = dict(
    backbone=dict(
        init_cfg=[
            dict(type='Xavier', layer='Conv2d', distribution='uniform'),
            dict(type='Xavier', layer='Conv1d', distribution='uniform'),
            dict(type='Xavier', layer='Linear', distribution='uniform'),
            dict(type='RNN', layer='LSTM', gain=1.0),
        ],
    ),
)
