"""Wave-3 retune: FastMLDNN @ RML2016.10A — Wave-2 combo + lr=5e-4.

Marginal peak lift (half of paper 4.4e-4) without destabilising overall.
"""

_base_ = [
    './wave2_fastmldnn_deepsig201610A_beta05_xavier_l2_dp007_esoff.py',
]

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=5e-4),
)
