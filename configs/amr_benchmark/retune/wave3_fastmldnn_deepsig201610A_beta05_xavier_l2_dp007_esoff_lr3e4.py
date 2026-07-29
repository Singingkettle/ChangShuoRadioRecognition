"""Wave-3 retune: FastMLDNN @ RML2016.10A — Wave-2 combo + lr=3e-4.

Siege round 2: between paper 4.4e-4 and lr5e4 (60.22%).
"""

_base_ = [
    './wave2_fastmldnn_deepsig201610A_beta05_xavier_l2_dp007_esoff.py',
]

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=3e-4),
)
