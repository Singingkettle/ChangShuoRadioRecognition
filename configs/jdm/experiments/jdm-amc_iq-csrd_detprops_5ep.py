# Bounded AMC fine-tune on detector-proposal crops.
#
# Closes the train/inference distribution gap: GT-box AMC training (~87% top1)
# vs joint inference on detector-localized crops. Uses a proposal cache from
# the optimized detector and fine-tunes from the baseline AMC checkpoint.
_base_ = '../jdm-amc_iq-csrd.py'

proposal_cache = 'work_dirs/jdm/amc_proposals/all_splits.json'

proposal_pipeline = [
    dict(type='LoadCSRDFrame'),
    dict(type='LoadDetProposal', proposal_cache=proposal_cache),
    dict(type='CSRDSignalToBaseband', source='frame'),
    dict(type='Reshape', shapes=dict(iq=[1, 2, 1200])),
    dict(type='PackInputs', input_key='iq'),
]

train_dataloader = dict(dataset=dict(pipeline=proposal_pipeline))
val_dataloader = dict(dataset=dict(pipeline=proposal_pipeline))
test_dataloader = dict(dataset=dict(pipeline=proposal_pipeline))

load_from = 'work_dirs/jdm/jdm-amc_iq-csrd/best_accuracy_top1_epoch_60.pth'

# Lower LR for short domain-adaptation fine-tune.
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=5e-5),
)

param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=5,
    eta_min=1e-6,
)

train_cfg = dict(by_epoch=True, max_epochs=5, val_interval=1)
work_dir = 'work_dirs/jdm/exp_amc_detprops_5ep'
