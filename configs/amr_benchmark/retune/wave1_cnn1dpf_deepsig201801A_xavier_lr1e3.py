"""Wave-1 retune: CNN1DPF @ RML2018.01A — Xavier init (Keras glorot_uniform)."""

_base_ = ['../../cnn1dpf/cnn1dpf_iq-deepsig-201801A.py']

model = dict(
    backbone=dict(
        init_cfg=[
            dict(type='Xavier', layer='Conv1d', distribution='uniform'),
            dict(type='Xavier', layer='Linear', distribution='uniform'),
        ],
    ),
)
