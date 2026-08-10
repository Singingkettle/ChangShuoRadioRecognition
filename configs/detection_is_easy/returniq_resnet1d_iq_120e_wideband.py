# Copyright (c) Shuo Chang and contributors. Licensed under the Apache License, Version 2.0.
# Return-to-IQ hierarchical recognizer on channelized wideband crops (input rep: iq).
# This is the recognition half of "Detection Is Easy, Recognition Is Hard": the
# detector supplies the boxes; this net re-classifies the IQ inside them. The 120-epoch
# AdamW + cosine + EMA + label-smoothing recipe is the budget lever that lifts accuracy.
# Paper: "Detection Is Easy, Recognition Is Hard: Rethinking Vision-Based Wideband Signal Detection and Recognition", IEEE TCCN (under review).
_base_ = ['../_base_/runtimes/amc.py']

# ----------------------------- data ------------------------------------------
# Point data_root at the directory holding the channelized caches produced by the
# return-to-IQ `build` step: train_L1024.npz / val_L1024.npz / test_L1024.npz,
# each with arrays X [N, 2, L] (float32) and y [N] (int64, 57 classes).
#
# This default is where `python tools/detection_is_easy/bridge.py build` writes them
# (bridge.py's CACHE root). If you moved the caches, override on the command line:
#   --cfg-options train_dataloader.dataset.data_root=<dir> #                 val_dataloader.dataset.data_root=<dir> #                 test_dataloader.dataset.data_root=<dir>
data_root = 'work_dirs/returniq_cache'
dataset_type = 'WidebandChannelizedDataset'

# The crop is already [2, L]; just pack it into the model input tensor.
pipeline = [dict(type='PackInputs', input_key='iq')]

train_dataloader = dict(
    batch_size=256,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train_L1024.npz',
        pipeline=pipeline,
        cache=True,
        test_mode=False),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=256,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val_L1024.npz',
        pipeline=pipeline,
        cache=True,
        test_mode=True),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = dict(
    batch_size=256,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test_L1024.npz',
        pipeline=pipeline,
        cache=True,
        test_mode=True),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

val_evaluator = dict(type='Accuracy', topk=(1,))
test_evaluator = val_evaluator

# ----------------------------- model -----------------------------------------
# The 12 OFDM (multi-carrier) classes; the coarse head routes them to the multi
# branch. These indices follow WIDEBAND_57_CLASSES order.
ofdm_indices = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]

model = dict(
    type='SignalClassifier',
    backbone=dict(
        type='ReturnIQResNet1D',
        input_rep='iq',
        stem_channels=64,
        stage_channels=(64, 128, 256),
        blocks_per_stage=2),
    head=dict(
        type='HierarchicalAMCHead',
        feat_dim=256,
        num_classes=57,
        multi_class_indices=ofdm_indices,
        label_smoothing=0.1,
        dropout=0.3),
)

# --------------------------- schedule (the recipe) ---------------------------
# 120 epochs, AdamW, linear warmup + cosine decay, EMA. The single knob that
# most moves the recognizer; a shorter schedule leaves accuracy on the table.
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=5e-2))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=True, begin=0, end=5,
         convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', by_epoch=True, begin=5, end=120, eta_min=1e-5,
         convert_to_iter_based=True),
]

train_cfg = dict(by_epoch=True, max_epochs=120, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# EMA over the recognizer weights; smooths the final model.
custom_hooks = [
    dict(type='EMAHook', ema_type='ExponentialMovingAverage', momentum=1e-4,
         update_buffers=True, priority=49),
]
