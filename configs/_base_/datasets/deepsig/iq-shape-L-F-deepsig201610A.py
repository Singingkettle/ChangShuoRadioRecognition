# dataset settings
data_root = 'data/ModulationClassification/DeepSig/RadioML.2016.10A'
dataset_type = 'AMCDataset'

# Per-sample L2 (unit-energy) normalization (see iq-deepsig201610A.py for the
# root-cause rationale): RML2016.10A ships each example at a tiny fixed scale
# (Frobenius ~0.1), which caps the temporal/RNN models well below the reference.
pipeline = [
    dict(
        type='SelfNormalize',
        norms=dict(iq={})
    ),
    dict(
        type='Transpose',
        orders=dict(iq=[1, 0])
    ),
    dict(
        type='PackInputs',
        input_key='iq'
    )
]

train_dataloader = dict(
    batch_size=400,
    num_workers=20,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.json',
        pipeline=pipeline,
        cache=True,
        test_mode=False),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=400,
    num_workers=20,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='validation.json',
        pipeline=pipeline,
        cache=True,
        test_mode=True),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

val_evaluator = [
    dict(type='Accuracy', topk=(1,)),
    dict(type='Loss', task='classification')
]

test_dataloader = dict(
    batch_size=400,
    num_workers=20,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        pipeline=pipeline,
        cache=True,
        test_mode=True),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
test_evaluator = dict(type='Accuracy', topk=(1,))
