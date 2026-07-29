"""Wave-1 retune: HCGDNN @ RML2016.10A — shared AMC schedule (lr=1e-3)."""

_base_ = ['../../hcgdnn/hcgdnn_iq-deepsig-201610A.py']

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-3),
)
