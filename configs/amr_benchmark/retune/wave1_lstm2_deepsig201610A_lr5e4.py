"""Wave-1 retune: LSTM2 @ RML2016.10A — lower LR for peak accuracy."""

_base_ = ['../../lstm2/lstm2_ap-shape-L-F-deepsig-201610A.py']

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=5e-4),
)
