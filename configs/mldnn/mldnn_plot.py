log_dir = '/home/citybuster/Data/SignalProcessing/Workdir'
legends = {
    'MLDNN': 0,
    'CLDNN-IQ': 1,
    'CLDNN-AP': 2,
    'CNN2-IQ': 3,
    'CNN2-AP': 4,
    'CNN3-IQ': 5,
    'CNN3-AP': 6,
    'CNN4-IQ': 7,
    'CNN4-AP': 8,
    'DensCNN-IQ': 9,
    'DensCNN-AP': 10,
    'ResCNN-IQ': 11,
    'ResCNN-AP': 12,
    'CGDNN2-IQ': 13,
    'CGDNN2-AP': 14,
    'CLDNN2-IQ': 15,
    'CLDNN2-AP': 16,
    'MLDNN-GRU': 20,
    'MLDNN-Last': 21,
    'MLDNN-Add': 22,
    'MLDNN-Att': 23,
    'MLDNN-Gradient': 24,
    'MLDNN-High': 25,
    'MLDNN-Low': 26,
    'DSCLDNN': 27,
    'MLDNN-IQ': 17,
    'MLDNN-AP': 18,
    'MLDNN-SNR': 19,
    'MLDNN-CNN': 28,
    'VGGNet-CO': 29,
    'AlexNet-CO': 30,
    'ResNet-CO': 31,
    'GoogleNet-CO': 32,
    'SVM-FB': 33,
    'DecisionTree-FB': 34,
}

plot = dict(
    type='MLDNNPlot',
    log_dir=log_dir,
    config='mldnn_mlnetv5_640_0.0004_0.5_deepsig_201610A',
    legends=legends,
    # Set the configs about confusion maps
    confusion_maps=[
        dict(
            type='ConfusionMap',
            log_dir=log_dir,
            name='confusion_map_mldnn_201610A.pdf',
            config='mldnn_mlnetv5_640_0.0004_0.5_deepsig_201610A',
            has_snr_classifier=True,
        ),
        dict(
            type='ConfusionMap',
            log_dir=log_dir,
            name='confusion_map_mldnn_201801A.pdf',
            config='mldnn_mlnetv5_640_0.0004_0.5_deepsig_201801A',
            has_snr_classifier=True,
        ),
    ],
    # Set the configs about training and test curves
    train_test_curves=[
        dict(
            type='SNRModulationCurve',
            name='cldnn_iq-ap.pdf',
            legends=legends,
            log_dir=log_dir,
            methods=[
                dict(
                    config='cldnn_deepsig_iq_201610A',
                    name='CLDNN-IQ',
                ),
                dict(
                    config='cldnn_deepsig_ap_201610A',
                    name='CLDNN-AP',
                ),
            ],
        ),
    ],
    # Set the configs about snr accuracy and modulation F1 score
    snr_modulation=[

        # Motivations
        dict(
            type='SNRModulationCurve',
            name='cldnn_iq-ap.pdf',
            legends=legends,
            log_dir=log_dir,
            methods=[
                dict(
                    config='cldnn_deepsig_iq_201610A',
                    name='CLDNN-IQ',
                ),
                dict(
                    config='cldnn_deepsig_ap_201610A',
                    name='CLDNN-AP',
                ),
            ],
        ),
        dict(
            type='SNRModulationCurve',
            name='cgdnn2_iq-ap.pdf',
            legends=legends,
            log_dir=log_dir,
            methods=[
                dict(
                    config='crdnn_gru_ap_deepsig_201610A',
                    name='CGDNN2-AP',
                ),
                dict(
                    config='crdnn_gru_iq_deepsig_201610A',
                    name='CGDNN2-IQ',
                ),
            ],
        ),
        dict(
            type='SNRModulationCurve',
            name='cldnn2_iq-ap.pdf',
            legends=legends,
            log_dir=log_dir,
            methods=[
                dict(
                    config='crdnn_lstm_ap_deepsig_201610A',
                    name='CLDNN2-AP',
                ),
                dict(
                    config='crdnn_lstm_iq_deepsig_201610A',
                    name='CLDNN2-IQ',
                ),
            ],
        ),

        # Ablation Study by an incremental way

        dict(
            type='SNRModulationCurve',
            name='ablation_data.pdf',
            legends=legends,
            log_dir=log_dir,
            methods=[
                # data, iq, ap
                dict(
                    config='cnn3_deepsig_iq_201610A',
                    name='CNN3-IQ',
                ),
                dict(
                    config='cnn3_deepsig_ap_201610A',
                    name='CNN3-AP',
                ),
                # iq+ap, with MLHead
                dict(
                    config='mldnn_mlnetv12_640_0.0004_0.5_deepsig_201610A',
                    name='MLDNN-CNN',
                    has_snr_classifier=True,
                ),
                # with BiGRU
                dict(
                    config='mldnn_mlnetv3_640_0.0004_0.5_deepsig_201610A_abl',
                    name='MLDNN-Last',
                    has_snr_classifier=True,
                ),
                # with SAFN
                dict(
                    config='mldnn_mlnetv11_640_0.0004_0.5_deepsig_201610A_abl',
                    name='MLDNN-Gradient',
                    has_snr_classifier=True,
                ),
                # without gradient pollution
                dict(
                    config='mldnn_mlnetv5_640_0.0004_0.5_deepsig_201610A',
                    name='MLDNN',
                    has_snr_classifier=True,
                ),

                # Other
                dict(
                    config='mldnn_mlnetv9_640_0.0004_0.5_deepsig_201610A_abl',
                    name='MLDNN-GRU',
                    has_snr_classifier=True,
                ),
                dict(
                    config='mldnn_mlnetv3_640_0.0004_0.5_deepsig_201610A_abl',
                    name='MLDNN-Last',
                    has_snr_classifier=True,
                ),
                dict(
                    config='mldnn_mlnetv4_640_0.0004_0.5_deepsig_201610A_abl',
                    name='MLDNN-Add',
                    has_snr_classifier=True,
                ),
                dict(
                    config='mldnn_mlnetv10_640_0.0004_0.5_deepsig_201610A_abl',
                    name='MLDNN-Att',
                    has_snr_classifier=True,
                ),
            ],
        ),

        # deepsig 201610A compare with I/Q A/P
        dict(
            type='SNRModulationCurve',
            name='deepsig_201610A_iq_ap.pdf',
            legends=legends,
            log_dir=log_dir,
            methods=[
                dict(
                    config='mldnn_mlnetv5_640_0.0004_0.5_deepsig_201610A',
                    name='MLDNN',
                    has_snr_classifier=True,
                ),
                dict(
                    config='dscldnn_deepsig_201610A',
                    name='DSCLDNN',
                ),
                dict(
                    config='rescnn_deepsig_iq_201610A',
                    name='ResCNN-IQ',
                ),
                dict(
                    config='cldnn_deepsig_iq_201610A',
                    name='CLDNN-IQ',
                ),
                dict(
                    config='cnn4_deepsig_iq_201610A',
                    name='CNN4-IQ',
                ),
                dict(
                    config='denscnn_deepsig_iq_201610A',
                    name='DensCNN-IQ',
                ),
            ],
        ),
        # deepsig 201601A compare with constellation and fb
        dict(
            type='SNRModulationCurve',
            name='deepsig_201610A_co_fb.pdf',
            legends=legends,
            log_dir=log_dir,
            methods=[
                dict(
                    config='mldnn_mlnetv5_640_0.0004_0.5_deepsig_201610A',
                    name='MLDNN',
                    has_snr_classifier=True,
                ),
                dict(
                    config='mldnn_alexnetco_640_0.0004_0.5_deepsig_201610A',
                    name='AlexNet-CO',
                ),
                dict(
                    config='mldnn_googlenetco_640_0.0004_0.5_deepsig_201610A',
                    name='GoogleNet-CO',
                ),
                dict(
                    config='mldnn_resnetco_640_0.0004_0.5_deepsig_201610A',
                    name='ResNet-CO',
                ),
                dict(
                    config='mldnn_vggnetco_640_0.0004_0.5_deepsig_201610A',
                    name='VGGNet-CO',
                ),
                dict(
                    config='mldnn_cul_dt_feature_based',
                    name='SVM-FB',
                ),
                dict(
                    config='mldnn_cul_svm_feature_based',
                    name='DecisionTree-FB',
                ),
            ],
        ),

        # deepsig 201801A compare with I/Q A/P
        dict(
            type='SNRModulationCurve',
            name='deepsig_201801A_iq_ap.pdf',
            legends=legends,
            log_dir=log_dir,
            methods=[
                dict(
                    config='mldnn_mlnetv5_640_0.0004_0.5_deepsig_201801A',
                    name='MLDNN',
                    has_snr_classifier=True,
                ),
                dict(
                    config='dscldnn_deepsig_201801A',
                    name='DSCLDNN',
                ),
                dict(
                    config='rescnn_deepsig_iq_201801A',
                    name='ResCNN-IQ',
                ),
                dict(
                    config='cldnn_deepsig_iq_201801A',
                    name='CLDNN-IQ',
                ),
                dict(
                    config='cnn4_deepsig_iq_201801A',
                    name='CNN4-IQ',
                ),
                dict(
                    config='denscnn_deepsig_iq_201801A',
                    name='DensCNN-IQ',
                ),
            ],
        ),
    ],
)
