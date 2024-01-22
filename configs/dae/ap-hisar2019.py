# dataset settings
data_root = 'data/ModulationClassification/Hisar/HisarMod2019.1'
dataset_type = 'AMCDataset'

pipeline = [
    dict(
        type='IQToAP',
    ),
    dict(
        type='DAENormalize'
    ),
    dict(
        type='Transpose',
        orders=dict(ap=[1, 0])
    ),
    dict(
        type='PackInputs',
        input_key='ap'
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
