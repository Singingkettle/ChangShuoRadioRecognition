# Wave-21 AMC: retrain on proposals from the BEST detector + box voting.
#
# Rationale (A1 finding): AMC top1 saturates ~83% and the wave-20 training-detail
# tricks (EMA + label smoothing + radio augment) slightly HURT (81.71% test).
# The strong-detector w17 crops came from det_full_60ep ep18 (ideal-det 0.693).
# The current best detector is det_full_120ep ep4 (ideal-det 0.759, and 0.824
# with box voting). Tighter, better-localized proposals -> cleaner baseband
# crops. This keeps the exact w17 recipe (cosine 60ep, ES) and ONLY swaps the
# proposal source; no tricks.
#
# PREREQUISITE: work_dirs/jdm/amc_proposals/det120ep_voted.json
#   built by tools/precompute_amc_proposals.py on det_full_120ep ep4 with
#   --cfg-options model.head.test_cfg.box_voting=True vote_iou_thr=0.75
_base_ = 'amc_detprops_strongdet_w17.py'

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
