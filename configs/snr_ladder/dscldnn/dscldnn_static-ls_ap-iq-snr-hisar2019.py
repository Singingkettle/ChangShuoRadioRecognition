# Static label smoothing route of the SNR-aware audit spectrum.
# Paper: "Do SNR-Aware Training Gains Survive a Frozen-Model Readout? A Null-Ladder Audit of Modulation Classification", under review (2026).
_base_ = ['dscldnn_hard-ce_ap-iq-snr-hisar2019.py']

work_dir = 'work_dirs/amc/hisar2019/dscldnn_static-ls'
method_name = 'static_ls_0p1'
experiment_note = 'HisarMod2019.1 DSCLDNN static label smoothing paired comparison after hard-CE baseline gate.'

model = dict(head=dict(loss=dict(
    type='LabelSmoothingCrossEntropyLoss',
    smoothing=0.1,
    loss_weight=1.0)))
