# Fair ideal-protocol detector eval: CSRD v1 only (no random impairments / infdB).
# User clarification 2026-07: Ideal = v1, not mixed simulate.
_base_ = '../../jdm-det_fft-csrd.py'

_ideal_versions = ['v1']

train_dataloader = dict(dataset=dict(versions=_ideal_versions))
val_dataloader = dict(dataset=dict(versions=_ideal_versions))
test_dataloader = dict(dataset=dict(versions=_ideal_versions))

work_dir = 'work_dirs/jdm/retune/eval_ideal_v1_det'
