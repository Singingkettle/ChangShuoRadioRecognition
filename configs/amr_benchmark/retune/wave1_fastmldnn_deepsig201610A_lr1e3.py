"""Wave-1 retune: FastMLDNN @ RML2016.10A — shared AMC lr (1e-3) vs paper 4.4e-4."""

_base_ = ['../../fastmldnn/fastmldnn_iq-ap-deepsig-201610A.py']

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-3),
)
