"""Wave-23: aug + SelfNormalize — denscnn@deepsig201610B (59.72/62; plain-aug w22 gave 57.35)."""
_base_ = ['./wave22_denscnn_deepsig201610B_radioaug_plateau.py']
train_pipeline = [
    dict(type='RadioAugment', key='iq', phase=True, time_shift=8, prob=0.9),
    dict(type='SelfNormalize', norms=dict(iq={})),
    dict(type='Reshape', shapes=dict(iq=[1, 2, 128])),
    dict(type='PackInputs', input_key='iq'),
]
test_pipeline = [
    dict(type='SelfNormalize', norms=dict(iq={})),
    dict(type='Reshape', shapes=dict(iq=[1, 2, 128])),
    dict(type='PackInputs', input_key='iq'),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
