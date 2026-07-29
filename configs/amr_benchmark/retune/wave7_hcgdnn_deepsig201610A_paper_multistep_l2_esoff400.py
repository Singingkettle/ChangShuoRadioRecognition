"""Wave-7 Tier-A: HCGDNN @ 10A — paper MultiStep 400ep + L2 (near-miss −2.17pp)."""

_base_ = ['./wave4_hcgdnn_deepsig201610A_paper_multistep_esoff400.py']

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.00044, weight_decay=1e-4),
)
