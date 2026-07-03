# CSRD (CRML23) single-signal baseband crops for the JDM modulation
# classification module. One sample per annotated signal: the frame is
# band-filtered around the ground-truth (center frequency, bandwidth) and
# shifted to baseband, mirroring the proposal filtering used at inference.
data_root = 'data/ChangShuo'
dataset_type = 'CSRDModulationDataset'

classes = ('16QAM', '64QAM', '8PSK', 'BPSK', 'QPSK')

pipeline = [
    dict(type='LoadCSRDFrame'),
    dict(type='CSRDSignalToBaseband', source='frame'),
    dict(type='Reshape', shapes=dict(iq=[1, 2, 1200])),
    dict(type='PackInputs', input_key='iq'),
]

train_dataloader = dict(
    batch_size=32,  # paper Sec. VI: classification batch size 32
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split='train',
        versions=None,
        metainfo=dict(classes=classes),
        pipeline=pipeline,
        test_mode=False),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=64,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split='validation',
        versions=None,
        metainfo=dict(classes=classes),
        pipeline=pipeline,
        test_mode=True),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

val_evaluator = [
    dict(type='Accuracy', topk=(1,)),
    dict(type='Loss', task='classification'),
]

test_dataloader = dict(
    batch_size=64,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split='test',
        versions=None,
        metainfo=dict(classes=classes),
        pipeline=pipeline,
        test_mode=True),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_evaluator = dict(type='Accuracy', topk=(1,))
