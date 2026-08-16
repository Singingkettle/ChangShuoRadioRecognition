# Static label smoothing route of the SNR-aware audit spectrum.
# Paper: "Do SNR-Aware Training Gains Survive a Frozen-Model Readout? A Null-Ladder Audit of Modulation Classification", under review (2026).
_base_ = ['../_base_/models/resnet_amr_iq-snr-deepsig-201801A.py']

work_dir = 'work_dirs/rcps_main_2018A/amc/deepsig201801A/resnet_amr_static-ls'
method_name = 'static_ls_0p1'

model = dict(
    head=dict(
        loss=dict(
            type='LabelSmoothingCrossEntropyLoss',
            smoothing=0.1,
            loss_weight=1.0)))

experiment_note = 'Static label smoothing placeholder for ResNet-AMR 2018A; use only after hard-CE gate passes.'
