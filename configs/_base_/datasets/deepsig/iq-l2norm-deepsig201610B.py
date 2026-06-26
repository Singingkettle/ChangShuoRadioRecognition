# RML2016.10B IQ pipeline with per-sample L2 (unit-energy) normalization.
# See iq-l2norm-deepsig201610A.py for the root-cause rationale (recurrent IQ
# models converge to a worse optimum at the tiny native input scale). Used by
# the recurrent IQ models (MCLDNN, CLDNNL, CLDNNW, CGDNet); the pure CNNs keep
# the un-normalized iq-deepsig201610B.py.
_base_ = ['./iq-deepsig201610B.py']

pipeline = [
    dict(type='SelfNormalize', norms=dict(iq={})),
    dict(type='Reshape', shapes=dict(iq=[1, 2, 128])),
    dict(type='PackInputs', input_key='iq'),
]

train_dataloader = dict(dataset=dict(pipeline=pipeline))
val_dataloader = dict(dataset=dict(pipeline=pipeline))
test_dataloader = dict(dataset=dict(pipeline=pipeline))
