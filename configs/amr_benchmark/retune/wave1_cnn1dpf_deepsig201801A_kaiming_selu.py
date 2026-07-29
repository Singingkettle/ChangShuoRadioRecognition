"""Wave-1 retune: CNN1DPF @ RML2018.01A — Kaiming fan-in for ReLU conv stack."""

_base_ = ['../../cnn1dpf/cnn1dpf_iq-deepsig-201801A.py']

model = dict(
    backbone=dict(
        init_cfg=[
            dict(type='Kaiming', layer='Conv1d', mode='fan_in', nonlinearity='relu'),
            dict(type='Xavier', layer='Linear', distribution='uniform'),
        ],
    ),
)

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=5e-4),
)
