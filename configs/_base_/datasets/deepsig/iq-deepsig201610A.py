# dataset settings
data_root = 'data/ModulationClassification/DeepSig/RadioML.2016.10A'
dataset_type = 'AMCDataset'

pipeline = [dict(type='Reshape', shapes=dict(iq=[1, 2, 128])), dict(type='PackInputs', input_key='iq')]

train_dataloader = dict(
    batch_size=400,
    num_workers=20,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='mm-train.json',
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
        ann_file='mm-validation.json',
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
        ann_file='mm-test.json',
        pipeline=pipeline,
        cache=True,
        test_mode=True),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
test_evaluator = dict(type='Accuracy', topk=(1,))
