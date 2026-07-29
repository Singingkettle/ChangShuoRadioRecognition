# AMC proposal-crop fine-tune on STRONG-DETECTOR proposals (architecture
# freeze). All previous AMC rungs (cosine 30/60ep, ft, plateau) trained on
# crops from the weak wave3b 5-ep detector and saturated at ~83.2 test top1
# (target 90). This regenerates the proposal cache with the current best
# detector (det_full_60ep ep18, ideal-det 0.8027) — better-localized crops,
# same model/pipeline/recipe.
# PREREQUISITE: work_dirs/jdm/amc_proposals/det_full_60ep_ep18.json
#   (built by tools/precompute_amc_proposals.py; see amc_strongdet_chain).
_base_ = 'amc_wave3b_detprops_30ep.py'

proposal_cache = 'work_dirs/jdm/amc_proposals/det_full_60ep_ep18.json'

# NOTE: must redefine the full pipeline — the base interpolates its own
# proposal_cache path at parse time, a bare variable override is a no-op.
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
