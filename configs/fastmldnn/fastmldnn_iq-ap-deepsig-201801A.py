_base_ = [
    './iq-ap-deepsig201801A.py',
    '../_base_/schedules/amc.py',
    '../_base_/runtimes/amc.py'
]

# model settings
model = dict(
    type='SignalClassifier',
    backbone=dict(
        type='FastMLDNN',
        num_classes=24,
    ),
    head=dict(
        type='FastMLDNNHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        beta=0,
    )
)

# ---------------------------------------------------------------------------
# RML2018.01A-specific optimisation stabilisation.
#
# Root cause of the previous garbage result (tested from epoch 1, overall 8.02%,
# peak 11.09%): with the shared ``_base_/schedules/amc.py`` learning rate of
# Adam 1e-3, FastMLDNN *diverged* on 2018.01A rather than failing to learn. The
# backbone merges its transformer sequence with ``merge='sum'`` (see
# csrr/models/backbones/fastmldnn.py); 2018.01A frames are 1024 samples long vs
# 128 on RML2016.10A/10B, so the post-CNN sequence is ~1018 vs ~122 steps and the
# summed feature magnitude -- hence logits and gradients -- is ~8x larger. At
# lr=1e-3 the first epoch's large update drove the final classifier ReLUs dead:
# the training loss locked at exactly ln(24)=3.1781 from epoch 2 onward and the
# validation accuracy collapsed to 1/24=4.17% (a constant-output network),
# leaving the epoch-1 weights as "best". RML2016.10A/10B are unaffected (much
# shorter signals -> ~8x smaller summed activations) and converge fine at the
# shared LR, so their configs are deliberately left untouched and this override
# is scoped to 2018.01A only.
#
# Fix (dataset-specific): lower the LR toward FastMLDNN's original paper value
# (4.4e-4) to 2e-4, add a short linear warmup so the early high-variance steps
# cannot kill the ReLUs, and clip the gradient norm as a hard guard against an
# explosive step. The LR then cosine-anneals over the remaining budget; the
# EarlyStoppingHook and max_epochs=150 from the inherited base still apply.
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=2e-4),
    clip_grad=dict(max_norm=5.0, norm_type=2),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=True, begin=0, end=5,
         convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', by_epoch=True, T_max=145, begin=5, end=150,
         eta_min=1e-6),
]
