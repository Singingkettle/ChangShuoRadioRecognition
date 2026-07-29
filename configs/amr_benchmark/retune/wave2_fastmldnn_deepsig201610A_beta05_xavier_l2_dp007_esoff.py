"""Wave-2 retune: FastMLDNN @ RML2016.10A — full P0+P1 stack (combo).

beta=0.5 + Xavier/TruncNormal + IQ L2 SelfNormalize + dp=0.07 + ES off.
Recommended single-shot combo from gap analysis fc5c869c / paper comparison §6.
"""

_base_ = ['../../fastmldnn/fastmldnn_iq-ap-deepsig-201610A.py']

_IQ_AP_L2_PIPELINE = [
    dict(type='SelfNormalize', norms=dict(iq={})),
    dict(type='MLDNNIQToAP'),
    dict(type='Reshape', shapes=dict(iq=[2, 128])),
    dict(type='Reshape', shapes=dict(ap=[2, 128])),
    dict(type='PackInputs', input_key=['iq', 'ap']),
]

train_dataloader = dict(dataset=dict(pipeline=_IQ_AP_L2_PIPELINE))
val_dataloader = dict(dataset=dict(pipeline=_IQ_AP_L2_PIPELINE))
test_dataloader = dict(dataset=dict(pipeline=_IQ_AP_L2_PIPELINE))

model = dict(
    backbone=dict(
        dp=0.07,
        init_cfg=[
            dict(type='Xavier', layer='Conv1d', distribution='uniform'),
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        ],
    ),
    head=dict(beta=0.5),
)

custom_hooks = []
