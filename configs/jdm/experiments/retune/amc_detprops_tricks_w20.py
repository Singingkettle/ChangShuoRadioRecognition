# Wave-20 AMC: narrative-safe training-detail tricks on strong-detector crops.
#
# AMC top1 has saturated at ~83% across every recipe tried (cosine 30/60ep,
# ReduceOnPlateau, strong-detector proposals). This run keeps the exact
# architecture / proposal source of amc_detprops_strongdet_w17 and adds three
# unpublished-but-benign training details that do not change the paper's
# narrative (single JDM classifier, same reported protocol):
#   - label smoothing 0.1 (softmax CE)
#   - EMA of the weights (evaluate the smoothed model)
#   - label-preserving radio augmentation (random carrier phase, small residual
#     CFO, symbol-timing roll) on the training crops only.
#
# PREREQUISITE (already on the H100):
#   work_dirs/jdm/amc_proposals/det_full_60ep_ep18.json
_base_ = 'amc_detprops_strongdet_w17.py'

proposal_cache = 'work_dirs/jdm/amc_proposals/det_full_60ep_ep18.json'

# Training pipeline = proposal pipeline + label-preserving augmentation
# inserted on the (2, L) baseband crop, BEFORE the Reshape to [1, 2, 1200].
train_pipeline = [
    dict(type='LoadCSRDFrame'),
    dict(type='LoadDetProposal', proposal_cache=proposal_cache),
    dict(type='CSRDSignalToBaseband', source='frame'),
    dict(type='RadioAugment', key='iq', phase=True, time_shift=8,
         freq_offset=0.01, prob=0.7),
    dict(type='Reshape', shapes=dict(iq=[1, 2, 1200])),
    dict(type='PackInputs', input_key='iq'),
]

# val/test keep the clean proposal pipeline (no augmentation) from w17.
eval_pipeline = [
    dict(type='LoadCSRDFrame'),
    dict(type='LoadDetProposal', proposal_cache=proposal_cache),
    dict(type='CSRDSignalToBaseband', source='frame'),
    dict(type='Reshape', shapes=dict(iq=[1, 2, 1200])),
    dict(type='PackInputs', input_key='iq'),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=eval_pipeline))
test_dataloader = dict(dataset=dict(pipeline=eval_pipeline))

# Label smoothing on the classification loss (softmax CE path only).
model = dict(
    head=dict(loss=dict(type='CrossEntropyLoss', loss_weight=1.0,
                        label_smoothing=0.1)))

# Exponential moving average of the weights, evaluated instead of the raw net.
custom_hooks = [
    dict(type='EMAHook', ema_type='ExponentialMovingAverage', momentum=0.0002,
         update_buffers=True, priority=49),
    dict(type='EarlyStoppingHook', monitor='accuracy/top1', min_delta=0,
         patience=25, rule='greater'),
]

param_scheduler = dict(
    type='CosineAnnealingLR', by_epoch=True, T_max=80, eta_min=1e-6)
train_cfg = dict(by_epoch=True, max_epochs=80, val_interval=1)

work_dir = 'work_dirs/jdm/retune/amc_detprops_tricks_w20'
