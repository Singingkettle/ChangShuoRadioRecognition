# RML2018.01A IQ pipeline with per-sample L2 (unit-energy) normalization.
# See iq-l2norm-deepsig201610A.py for the root-cause rationale. Length-1024
# frames. Used by the recurrent IQ models (MCLDNN, CLDNNL, CLDNNW, CGDNet);
# the pure CNNs keep the un-normalized iq-deepsig201801A.py.
_base_ = ['./iq-deepsig201801A.py']

pipeline = [
    dict(type='SelfNormalize', norms=dict(iq={})),
    dict(type='Reshape', shapes=dict(iq=[1, 2, 1024])),
    dict(type='PackInputs', input_key='iq'),
]

train_dataloader = dict(dataset=dict(pipeline=pipeline))
val_dataloader = dict(dataset=dict(pipeline=pipeline))
test_dataloader = dict(dataset=dict(pipeline=pipeline))
