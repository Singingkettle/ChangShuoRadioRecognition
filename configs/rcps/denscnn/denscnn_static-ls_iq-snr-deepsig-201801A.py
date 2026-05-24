_base_ = ['../_base_/models/denscnn_iq-snr-deepsig-201801A.py']

work_dir = '/home/citybuster/Data/RCPS/work_dirs/rcps_main_2018A/amc/deepsig201801A/denscnn_static-ls'
method_name = 'static_ls_0p1'

model = dict(
    head=dict(
        loss=dict(
            type='LabelSmoothingCrossEntropyLoss',
            smoothing=0.1,
            loss_weight=1.0)))

experiment_note = 'Static label smoothing placeholder for DenseNet-AMR/DensCNN 2018A; use only after hard-CE gate passes.'
