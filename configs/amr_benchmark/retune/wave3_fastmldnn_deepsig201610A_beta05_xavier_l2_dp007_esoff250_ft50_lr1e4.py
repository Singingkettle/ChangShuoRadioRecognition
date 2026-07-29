"""Wave-3 retune: FastMLDNN @ RML2016.10A — fine-tune from esoff250 best.

Siege round 3: load esoff250 best (60.90/91.18), 50 more epochs at lr=1e-4.
"""

_base_ = [
    './wave2_fastmldnn_deepsig201610A_beta05_xavier_l2_dp007_esoff.py',
]

load_from = (
    'work_dirs/amr_benchmark_retune/fastmldnn/deepsig201610A/'
    'beta05_xavier_l2_dp007_esoff250/best_accuracy_top1_epoch_232.pth')

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-4),
)

train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=1)

param_scheduler = dict(
    _delete_=True,
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=50,
    eta_min=1e-6,
)
