# AMC fine-tune on det120 + box-voting proposals (ideal-joint operating point).
# Same recipe as amc_detprops_strongdet_w17.py; only the proposal source changes.
# Prerequisite: configs/jdm/scripts/precompute_amc_proposals.py on det_full_120ep with
#   --cfg-options model.head.test_cfg.box_voting=True vote_iou_thr=0.75
# Paper: Xing et al., IEEE TWC 2024.
_base_ = './amc_detprops_strongdet_w17.py'

proposal_cache = 'work_dirs/jdm/amc_proposals/det120ep_voted.json'

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

work_dir = 'work_dirs/jdm/retune/amc_detprops_120voted_w21'
