"""Wave-22: stronger augmentation — mcnet@deepsig201610A (57.04/58)."""
_base_ = ['./wave20_mcnet_deepsig201610A_radioaug_plateau.py']
train_pipeline = [
    dict(type='RadioAugment', key='iq', phase=True, time_shift=16,
         freq_offset=0.005, prob=1.0),
    dict(type='Reshape', shapes=dict(iq=[1, 2, 128])),
    dict(type='PackInputs', input_key='iq'),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
