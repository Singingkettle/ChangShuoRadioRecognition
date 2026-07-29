# P1 AMC: 30-ep proposal-crop fine-tune from 20-ep best, using Track B
# detector proposals (det_wave3b_5ep_lr1e3, test mAP 0.8113).
# Experimental only — do not replace production AMC until joint/AP75 review.
_base_ = '../../experiments/jdm-amc_iq-csrd_detprops_20ep.py'

proposal_cache = 'work_dirs/jdm/amc_proposals/wave3b_5ep_lr1e3.json'

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

load_from = 'work_dirs/jdm/exp_amc_detprops_20ep/best_accuracy_top1_epoch_20.pth'

param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=30,
    eta_min=1e-6,
)

train_cfg = dict(by_epoch=True, max_epochs=30, val_interval=1)
work_dir = 'work_dirs/jdm/retune/amc_wave3b_detprops_30ep'
