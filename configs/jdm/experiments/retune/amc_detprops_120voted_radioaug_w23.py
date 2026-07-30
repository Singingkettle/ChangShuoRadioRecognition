# Wave-23 AMC: w21 recipe (box-voted det120 proposals) + RadioAugment ONLY.
#
# Motivation: w21 (84.63% test) beats w17 (83.26%) on aggregate top1 and lifts
# ideal joint mAP (0.7667) but HURTS simulate joint mAP (0.4485 vs 0.5195) —
# it was trained on cleaner voted crops and became brittle on noisy real_awgn
# crops. W20 bundled three tricks (EMA + label smoothing + augment) and lost
# aggregate accuracy, so the confounders are removed here: keep the exact w21
# recipe and add ONLY the label-preserving radio augmentation to make the
# classifier robust to crop nuisances (phase/CFO/timing). Target metric is the
# SIMULATE joint mAP after merging with det120.
_base_ = 'amc_detprops_120voted_w21.py'

proposal_cache = 'work_dirs/jdm/amc_proposals/det120ep_voted.json'

train_pipeline = [
    dict(type='LoadCSRDFrame'),
    dict(type='LoadDetProposal', proposal_cache=proposal_cache),
    dict(type='CSRDSignalToBaseband', source='frame'),
    dict(type='RadioAugment', key='iq', phase=True, time_shift=8,
         freq_offset=0.01, prob=0.7),
    dict(type='Reshape', shapes=dict(iq=[1, 2, 1200])),
    dict(type='PackInputs', input_key='iq'),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

work_dir = 'work_dirs/jdm/retune/amc_detprops_120voted_radioaug_w23'
