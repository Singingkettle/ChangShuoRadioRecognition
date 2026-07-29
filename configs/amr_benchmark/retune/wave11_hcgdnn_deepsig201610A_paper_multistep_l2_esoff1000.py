"""Wave-11 Tier-A: HCGDNN MultiStep + L2, 1000ep (longer budget). Arch freeze."""
_base_ = ['./wave7_hcgdnn_deepsig201610A_paper_multistep_l2_esoff400.py']
train_cfg = dict(by_epoch=True, max_epochs=1000, val_interval=1)
# Keep MultiStep from base; extend early-stop off via long max_epochs.
work_dir = (
    'work_dirs/amr_benchmark_retune/hcgdnn/deepsig201610A/'
    'paper_multistep_l2_esoff1000_w11')
