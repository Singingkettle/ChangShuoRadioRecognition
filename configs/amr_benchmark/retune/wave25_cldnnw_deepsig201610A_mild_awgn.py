"""Wave-25: mild phase + AWGN injection — cldnnw@deepsig201610A (54.73 mild / 57 target)."""
_base_ = ['./wave24_cldnnw_deepsig201610A_radioaug_mild.py']
train_pipeline = [
    dict(type='RadioAugment', key='iq', phase=True, time_shift=0,
         awgn_snr_db=(0, 30), awgn_prob=0.4, prob=0.5),
    dict(type='SelfNormalize', norms=dict(iq={})),
    dict(type='Reshape', shapes=dict(iq=[1, 2, 128])),
    dict(type='PackInputs', input_key='iq'),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
