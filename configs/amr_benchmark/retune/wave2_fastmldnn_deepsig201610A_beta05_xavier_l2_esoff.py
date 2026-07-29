"""Wave-2 retune: FastMLDNN @ RML2016.10A — P0 + P1 IQ L2 SelfNormalize.

Same as beta05_xavier_esoff150 plus per-sample L2 norm on IQ before
MLDNNIQToAP (MCLDNN success path; see audit_changes.md §RNN ceiling).
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
        init_cfg=[
            dict(type='Xavier', layer='Conv1d', distribution='uniform'),
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        ],
    ),
    head=dict(beta=0.5),
)

custom_hooks = []
