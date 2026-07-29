"""Wave-7 Tier-A: HCGDNN @ 10A — paper MultiStep 400ep @ lr=3e-4."""

_base_ = ['./wave4_hcgdnn_deepsig201610A_paper_multistep_esoff400.py']

optim_wrapper = dict(optimizer=dict(type='Adam', lr=3e-4))
