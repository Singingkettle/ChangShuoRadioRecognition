"""Wave-12 Tier-A: FastMLDNN author-style STAGE-2 fine-tune from current best.

Upstream stage-2 (fastmldnn_stage2_iq-ap-deepsig-201610A.py): resume the best
stage-1 backbone, then fine-tune with dp=0.07 + beta=0.5 at CONSTANT lr
1.054e-4 for a very long budget. We apply that exact stage-2 recipe to our
best available checkpoint (wave-3 beta05_xavier_l2_dp007_esoff300, test 61.02,
already trained with beta=0.5/dp=0.07/L2 pipeline — pipeline kept identical to
match the checkpoint's input scale). Patient ES (min_delta=0, patience=100)
mirrors the author's tolerance instead of our earlier tight ES.
"""
_base_ = ['./wave2_fastmldnn_deepsig201610A_beta05_xavier_l2_dp007_esoff.py']

load_from = (
    'work_dirs/amr_benchmark_retune/fastmldnn/deepsig201610A/'
    'beta05_xavier_l2_dp007_esoff300/best_accuracy_top1_epoch_214.pth')

optim_wrapper = dict(optimizer=dict(type='Adam', lr=1.054e-4))

param_scheduler = dict(_delete_=True, type='ConstantLR', factor=1)

train_cfg = dict(by_epoch=True, max_epochs=1600, val_interval=1)

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='accuracy/top1', min_delta=0,
         patience=100, rule='greater'),
]
