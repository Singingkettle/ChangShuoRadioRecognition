log_dir = '/home/citybuster/Data/SignalProcessing/Workdir'
legends = {
    'FMLDNN-bv5': 0,
    'FMLDNN-bv6': 1,
}

plot = dict(
    type='MLDNNPlot',
    log_dir=log_dir,
    config='fmldnn',
    legends=legends,
    # Set the configs about confusion maps
    confusion_maps=[
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201610A.pdf',
        #     config='fmldnn_fmlnetv5_640_0.0004_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv6_640_0.0004_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv6_640_0.0004_0.1_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv6_640_0.0004_0.2_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv6_640_0.0004_0.3_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv6_640_0.0004_0.4_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),

        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv6_160_0.0004_0.4_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv6_320_0.0004_0.4_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv6_640_0.0010_0.4_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv6_640_0.0020_0.4_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv6_320_0.0001_0.4_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv6_320_0.0002_0.4_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv6_sgd_320_0.0004_0.4_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv6_nhead-4_320_0.0004_0.4_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv6_nhead-8_320_0.0004_0.4_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),

        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv7_nhead-4_320_0.0004_0.4_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv8_nhead-4_320_0.0004_0.4_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv9_nhead-4_320_0.0004_0.4_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv10_nhead-4_320_0.0004_0.4_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv6_nlayers-3_nhead-4_320_0.0004_0.4_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),

        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv11_nhead-4_320_0.0004_0.4_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv12_640_0.0004_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv13_640_0.0004_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv14_640_0.0004_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),

        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv15_640_0.0004_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv16_640_0.0004_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv17_640_0.0004_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv18_nipn-16_640_0.0004_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv18_nipn-32_640_0.0004_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv18_nipn-64_640_0.0004_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),

        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv19_640_0.0004_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv20_640_0.0004_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv21_640_0.0004_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv22_640_0.0004_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv23_640_0.0004_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),

        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv20_640_0.0006_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv20_640_0.0007_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv20_640_0.0008_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv20_640_0.0009_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv20_640_0.0010_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv20_640_rmsprop_0.0004_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv20_640_adagrad_0.0004_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv20_640_admw_0.0004_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),

        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv24_640_0.0004_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv25_640_0.0004_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        #

        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv27_640_0.0010_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv28_640_0.0010_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv29_640_0.0010_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),

        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv30_640_0.0002_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv30_640_0.0006_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv30_640_0.0010_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),

        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv31_640_0.0010_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv31_640_0.0020_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv32_640_0.0010_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),
        # dict(
        #     type='ConfusionMap',
        #     log_dir=log_dir,
        #     name='confusion_map_fmldnn_201801A.pdf',
        #     config='fmldnn_fmlnetv32_640_0.0020_0.5_deepsig_201610A',
        #     has_snr_classifier=True,
        # ),

        dict(
            type='ConfusionMap',
            log_dir=log_dir,
            name='confusion_map_fmldnn_201801A.pdf',
            config='fmldnn_fmlnetv33_640_0.0010_0.5_deepsig_201610A',
            has_snr_classifier=True,
        ),
        dict(
            type='ConfusionMap',
            log_dir=log_dir,
            name='confusion_map_fmldnn_201801A.pdf',
            config='fmldnn_fmlnetv33_640_0.0015_0.5_deepsig_201610A',
            has_snr_classifier=True,
        ),
        dict(
            type='ConfusionMap',
            log_dir=log_dir,
            name='confusion_map_fmldnn_201801A.pdf',
            config='fmldnn_fmlnetv34_640_0.0010_0.5_deepsig_201610A',
            has_snr_classifier=True,
        ),
        dict(
            type='ConfusionMap',
            log_dir=log_dir,
            name='confusion_map_fmldnn_201801A.pdf',
            config='fmldnn_fmlnetv34_640_0.0015_0.5_deepsig_201610A',
            has_snr_classifier=True,
        ),
    ],
)