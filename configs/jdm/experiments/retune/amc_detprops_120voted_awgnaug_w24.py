# Wave-24 AMC: w21 recipe + AWGN-injection augmentation for SIMULATE robustness.
#
# Motivation: every classifier trained on (clean) voted det120 crops improves
# aggregate top1 (w21 84.63, w23 84.69) and ideal joint mAP, but LOSES on the
# simulate (real/real_awgn) joint protocol vs the older w17 model
# (0.4485/0.4510 vs 0.5195) — the classifiers never see noisy crops during
# training. w23's phase/CFO/timing augmentation alone did not fix it, because
# the dominant simulate impairment is additive noise. This run keeps the w21
# recipe and adds random-SNR AWGN injection (plus the mild w23 nuisances) on
# the training crops only. AWGN injection is label-preserving and exactly the
# channel model of the paper's own AWGN/Real_awgn versions — narrative-safe.
# Target metric: simulate joint mAP after merging with det120 (floor 0.67).
_base_ = 'amc_detprops_120voted_w21.py'

proposal_cache = 'work_dirs/jdm/amc_proposals/det120ep_voted.json'

train_pipeline = [
    dict(type='LoadCSRDFrame'),
    dict(type='LoadDetProposal', proposal_cache=proposal_cache),
    dict(type='CSRDSignalToBaseband', source='frame'),
    dict(type='RadioAugment', key='iq', phase=True, time_shift=8,
         freq_offset=0.01, awgn_snr_db=(-6, 30), awgn_prob=0.5, prob=0.9),
    dict(type='Reshape', shapes=dict(iq=[1, 2, 1200])),
    dict(type='PackInputs', input_key='iq'),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

work_dir = 'work_dirs/jdm/retune/amc_detprops_120voted_awgnaug_w24'
