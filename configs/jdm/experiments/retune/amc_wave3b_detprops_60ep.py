# AMC proposal-crop fine-tune, 60-epoch cosine (architecture freeze; longer
# schedule than the 30ep rung) to probe the AMC proposal top1 gap (83% -> 90%).
# SAME model/data/pipeline as amc_wave3b_detprops_30ep.py; only the training
# LENGTH (schedule + max_epochs) changes.
#
# PREREQUISITES (must exist before this can train — the escalation ladder in
# tools/amr_benchmark/gpu_pool_keepalive.sh guards on the proposal cache and
# will SKIP this rung if the cache is absent, to avoid crash-churn):
#   work_dirs/jdm/amc_proposals/wave3b_5ep_lr1e3.json   (proposal_cache)
#   work_dirs/jdm/exp_amc_detprops_20ep/best_accuracy_top1_epoch_20.pth (load_from)
_base_ = 'amc_wave3b_detprops_30ep.py'

param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=60,
    eta_min=1e-6,
)

train_cfg = dict(by_epoch=True, max_epochs=60, val_interval=1)

work_dir = 'work_dirs/jdm/retune/amc_wave3b_detprops_60ep'
