"""Wave-27: resnetamr@deepsig201610B (60.22/pass 60.5) — mild phase-only augment.

Full augment (phase+shift8 p=0.9) plateaued at 60.22 across two seeds; the
mild direction (phase-only p=0.5) is what unlocked cldnnw-10A.
"""
_base_ = ['./wave22_resnetamr_deepsig201610B_radioaug_plateau.py']
train_pipeline = [
    dict(type='RadioAugment', key='iq', phase=True, time_shift=0, prob=0.5),
    dict(type='Reshape', shapes=dict(iq=[1, 2, 128])),
    dict(type='PackInputs', input_key='iq'),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
