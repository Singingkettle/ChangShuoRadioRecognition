"""Wave-4 marginal retune: HCGDNN @ RML2016.10A — lower LR for overall lift."""

_base_ = ['../../hcgdnn/hcgdnn_iq-deepsig-201610A.py']

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=5e-4),
)
