# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

_base_ = './rtmdet_m_stft_3ch_resize512.py'

# Axis-A (input representation): raw-IQ -> in-graph FROZEN-Fourier filterbank -> [real, imag, log-mag],
# on the hardshort-lowsnr 65k raw IQ, 57-class via --root coco_multiclass. This is the A5 (frozen-Fourier)
# cell and the BASE for the phase-ablation: channel_mode='realimag_logmag' (phase IN) vs a magnitude-only
# variant (phase OUT). apply_tensor_stats() skips RawIQFilterbank preprocessors, so these mean/std stand.
# NOTE ON THE NORMALISATION STATISTICS: mean/std below were carried over from the offline STFT3
# statistics rather than recomputed on this front end's own output. They started as a placeholder,
# but every reported number for this cell (and for the magnitude-only cell it is the base of) was
# trained with exactly these values, so they are part of the published recipe. Do not "fix" them
# when reproducing -- recomputing them changes the phase-test comparison. The same constants are
# used by both arms of that test, so the comparison itself stays fair.
raw_root = 'data/torchsig_hardshort_lowsnr_iq_65k_nvme/raw'

model = dict(
    data_preprocessor=dict(
        _delete_=True,
        type='RawIQFilterbankDetDataPreprocessor',
        mean=[-0.00005, -0.000306, 0.221728],
        std=[0.96207, 0.962656, 0.330376],
        pad_size_divisor=32,
        num_bins=512,
        kernel_size=513,
        stride=512,
        filterbank_init='fourier',
        filterbank_window='hann',
        filterbank_type='gabor',
        channel_mode='realimag_logmag',
        trainable_filterbank=False,
        output_size=(512, 512),
    ),
    backbone=dict(input_channels=3),
)

train_pipeline = [
    dict(type='LoadRawIQFromCOCOStem', raw_root=raw_root, target_shape=(512, 512)),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadRawIQFromCOCOStem', raw_root=raw_root, target_shape=(512, 512)),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'raw_iq_shape', 'raw_iq_path'),
    ),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

work_dir = 'work_dirs/torchsig_mmdet_raw_iq_filterbank_hardshort_rtmdet_m'
