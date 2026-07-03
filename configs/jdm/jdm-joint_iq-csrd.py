# End-to-end JDM evaluation: detector proposals -> baseband filtering ->
# modulation classification, scored with class-aware detection mAP.
#
# This config is inference-only. Train the two modules first with
# ``jdm-det_fft-csrd.py`` and ``jdm-amc_iq-csrd.py``, then either point
# ``load_from`` to a merged checkpoint or set the per-submodule Pretrained
# ``init_cfg`` entries below to the trained checkpoints, e.g.::
#
#     model = dict(detector=dict(init_cfg=dict(
#         type='Pretrained', checkpoint='work_dirs/jdm-det_fft-csrd/best.pth')))
#
# Run with: python tools/test_det.py configs/jdm/jdm-joint_iq-csrd.py <ckpt>
_base_ = [
    '../_base_/datasets/csrd/det-fft-csrd.py',
    '../_base_/runtimes/det.py',
]

model = dict(
    type='JDMFramework',
    detector=dict(
        type='SignalDetector',
        backbone=dict(
            type='JDMDetectionBackbone',
            in_channels=2,
            stage_channels=(16, 32, 64, 128, 256),
        ),
        head=dict(
            type='JDMDetectionHead',
            in_channels=256,
            frame_length=1200,
            stride=8,
            anchor_widths=(100.0, 120.0, 140.0),
            test_cfg=dict(score_thr=0.05, nms_iou_thr=0.45, max_per_frame=20),
        ),
    ),
    classifier=dict(
        type='SignalClassifier',
        backbone=dict(
            type='JDMClassificationBackbone',
            num_classes=5,
            dropout_rate=0.5,
        ),
        head=dict(
            type='ClsHead',
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        ),
    ),
)

# the framework consumes time-domain frames and computes the FFT internally,
# so drop the IQToSpectrum stage of the detection pipeline
joint_pipeline = [
    dict(type='LoadCSRDFrame'),
    dict(type='PackDetectionInputs', input_key='iq'),
]

val_dataloader = dict(dataset=dict(pipeline=joint_pipeline))
test_dataloader = dict(dataset=dict(pipeline=joint_pipeline))

# class-aware evaluation over modulation labels = joint JDM metric
val_evaluator = dict(type='SignalDetectionMetric', classwise=True)
test_evaluator = dict(type='SignalDetectionMetric', classwise=True)

# inference-only: no train loop
train_dataloader = None
train_cfg = None
optim_wrapper = None
param_scheduler = None
val_cfg = dict()
test_cfg = dict()
