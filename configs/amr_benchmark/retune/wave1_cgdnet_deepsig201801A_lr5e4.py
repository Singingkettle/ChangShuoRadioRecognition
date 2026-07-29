"""Wave-1 retune: CGDNet @ RML2018.01A — moderate LR halving."""

_base_ = ['../../cgdnet/cgdnet_iq-deepsig-201801A.py']

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=5e-4),
)
