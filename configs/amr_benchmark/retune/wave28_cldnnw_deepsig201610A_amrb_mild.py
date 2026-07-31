"""Wave-28: cldnnw@deepsig201610A — amrb recipe + mild phase augment.

amrb_plateau_w15 is the pair's real best: overall 55.24 (pass 55.5, -0.26)
but peak 81.95 (pass 84.0, -2.05). Mild phase augment lifted peak on sibling
pairs; combine it with the winning amrb schedule to fix both axes.
"""
_base_ = ['./wave15_cldnnw_deepsig201610A_amrb_plateau.py']
train_pipeline = [
    dict(type='RadioAugment', key='iq', phase=True, time_shift=0, prob=0.5),
    dict(type='Reshape', shapes=dict(iq=[1, 2, 128])),
    dict(type='PackInputs', input_key='iq'),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
