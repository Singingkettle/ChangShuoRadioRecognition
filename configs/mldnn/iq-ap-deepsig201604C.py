# dataset settings
data_root = 'data/ModulationClassification/DeepSig/RadioML.2016.04C'
dataset_type = 'AMCDataset'

train_pipeline = [
    dict(
        type='MLDNNSNRLabel'
    ),
    dict(
        type='MLDNNIQToAP',
    ),
    dict(
        type='Reshape',
        shapes=dict(iq=[1, 2, 128])
    ),
    dict(
        type='Reshape',
        shapes=dict(ap=[1, 2, 128])
    ),
    dict(
        type='PackMultiTaskInputs',
        multi_task_fields=['gt_label'],
        input_key=['iq', 'ap'],
        task_handlers=dict(
            amc=dict(
                type='PackInputs'
            ),
            snr=dict(
                type='PackInputs'
            ),
        ),
    )
]

pipeline = [
    dict(
        type='MLDNNIQToAP',
    ),
    dict(
        type='Reshape',
        shapes=dict(iq=[1, 2, 128])
    ),
    dict(
        type='Reshape',
        shapes=dict(ap=[1, 2, 128])
    ),
    dict(
        type='PackInputs',
        input_key=['iq', 'ap'],
    )
]

train_dataloader = dict(
    batch_size=640,
    num_workers=20,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.json',
        pipeline=train_pipeline,
        cache=True,
        test_mode=False),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=640,
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
    batch_size=640,
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
