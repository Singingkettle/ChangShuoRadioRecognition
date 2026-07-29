# CSRD (CRML23) multi-signal frames, frequency-domain input for the JDM
# signal-detection module. Frames are the stored wideband_data (received
# frame, noise applied once); the detector consumes the fftshift-ed FFT
# (amplitude + phase).
data_root = '/home/citybuster/Data/WirelessRadio/data/ChangShuoTwc2026'
dataset_type = 'CSRDDetectionDataset'

# Deterministic per-version 50/10/40 split (repo convention); no split files
# exist on disk. Modulation order is pinned for reproducibility.
classes = ('16QAM', '64QAM', '8PSK', 'BPSK', 'QPSK')

pipeline = [
    dict(type='LoadCSRDFrame'),
    dict(type='IQToSpectrum'),
    dict(type='PackDetectionInputs', input_key='spectrum'),
]

train_dataloader = dict(
    batch_size=12,  # paper Sec. VI: detection batch size 12
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split='train',
        versions=None,  # all v* directories
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

val_evaluator = dict(type='SignalDetectionMetric')

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

test_evaluator = dict(type='SignalDetectionMetric')
