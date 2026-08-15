# AMC fine-tune on detector-proposal crops (simulate-joint operating point).
# Starts from the GT-box AMC checkpoint; 60-epoch cosine at 1e-4.
# Prerequisite: configs/jdm/scripts/precompute_amc_proposals.py → the proposal_cache below.
# Paper: Xing et al., IEEE TWC 2024.
_base_ = '../jdm-amc_iq-csrd.py'

proposal_cache = 'work_dirs/jdm/amc_proposals/det_full_60ep_ep18.json'

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

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=5e-5),
)

param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=60,
    eta_min=1e-6,
)

train_cfg = dict(by_epoch=True, max_epochs=60, val_interval=1)

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='accuracy/top1', min_delta=0,
         patience=25, rule='greater'),
]

work_dir = 'work_dirs/jdm/retune/amc_detprops_strongdet_w17'
