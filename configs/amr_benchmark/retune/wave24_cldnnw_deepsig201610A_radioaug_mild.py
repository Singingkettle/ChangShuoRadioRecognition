"""Wave-24: MILD augment (phase-only, p=0.5) — cldnnw@deepsig201610A (53.78/57; stronger augs hurt)."""
_base_ = ['./wave20_cldnnw_deepsig201610A_radioaug_plateau.py']
train_pipeline = [
    dict(type='RadioAugment', key='iq', phase=True, time_shift=0, prob=0.5),
    dict(type='SelfNormalize', norms=dict(iq={})),
    dict(type='Reshape', shapes=dict(iq=[1, 2, 128])),
    dict(type='PackInputs', input_key='iq'),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
