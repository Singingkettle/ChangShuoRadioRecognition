log_dir = '/home/citybuster/Data/SignalProcessing/Workdir'
legend = {
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
    'Fast MLDNN': 35,
    'Fast MLDNN-CE': 36,
    'Fast MLDNN-CE-CM': 37,
    'Fast MLDNN-FL-CM': 38,
    'Fast MLDNN-FL-CM-EP': 39,
    'Fast MLDNN-CE-CM-CN': 40,
    'Fast MLDNN-FL-CM-CN': 41,
    'Fast MLDNN-FL-CM-CN-EP': 42,
    'Fast MLDNN-CE-CM-CN-EP': 43,
    'HCGDNN': 44,
    'HCGDNN-CNN': 45,
    'HCGDNN-GRU1': 46,
    'HCGDNN-GRU2': 47,
    'HCGDNN-CG1': 48,
    'HCGDNN-CG2': 49,
}

plot = dict(
    type='CommonPlot',
    log_dir=log_dir,
    config='hcgdnn_hcgnetv2_100_288_lr-0.00044-step-800_deepsig_iq_201610A',
    legend=legend,

    # Set the configs about confusion maps
    confusion_map=[
        dict(
            type='ConfusionMap',
            log_dir=log_dir,
            name='confusion_map_201610A.pdf',
            method=dict(
                config='hcgdnn_hcgnetv2_100_288_lr-0.00044-step-800_deepsig_iq_201610A',
                name='HCGDNN',
            ),
        ),
    ],

    # Set the configs about snr accuracy and modulation F1 score
    snr_modulation=[
        # Ablation Study by an incremental way
        dict(
            type='AccuracyF1Plot',
            name='ablation_data.pdf',
            legend=legend,
            log_dir=log_dir,
            method=[
                # CNN
                dict(
                    config='hcgdnn_abl_cnn_deepsig_iq_201610A',
                    name='HCGDNN-CNN',
                ),
                # GRU1
                dict(
                    config='hcgdnn_abl_gru1_deepsig_iq_201610A',
                    name='HCGDNN-GRU1',
                ),
                # GRU2
                dict(
                    config='hcgdnn_abl_gru2_deepsig_iq_201610A',
                    name='HCGDNN-GRU2',
                ),
                # CNN+GRU1
                dict(
                    config='hcgdnn_abl_cg1_deepsig_iq_201610A',
                    name='HCGDNN-CG1',
                ),
                # CNN+GRU2
                dict(
                    config='hcgdnn_abl_cg2_deepsig_iq_201610A',
                    name='HCGDNN-CG2',
                ),
                # GRU1+GRU2
                dict(
                    config='hcgdnn_abl_g1G2_deepsig_iq_201610A',
                    name='HCGDNN-G1G2',
                ),
                # Final Version
                dict(
                    config='hcgdnn_hcgnetv2_100_288_lr-0.00044-step-800_deepsig_iq_201610A',
                    name='HCGDNN',
                ),
            ],
        ),

        # deepsig 201610A for cnn gru1
        dict(
            type='AccuracyF1Plot',
            name='motivation_cg1_deepsig_201610A.pdf',
            legend=legend,
            log_dir=log_dir,
            method=[
                # CNN
                dict(
                    config='hcgdnn_abl_cnn_deepsig_iq_201610A',
                    name='HCGDNN-CNN',
                ),
                # GRU1
                dict(
                    config='hcgdnn_abl_gru1_deepsig_iq_201610A',
                    name='HCGDNN-GRU1',
                ),
            ],
        ),
        # deepsig 201610A for cnn gru2
        dict(
            type='AccuracyF1Plot',
            name='motivation_cg2_deepsig_201610A.pdf',
            legend=legend,
            log_dir=log_dir,
            method=[
                # CNN
                dict(
                    config='hcgdnn_abl_cnn_deepsig_iq_201610A',
                    name='HCGDNN-CNN',
                ),
                # GRU2
                dict(
                    config='hcgdnn_abl_gru2_deepsig_iq_201610A',
                    name='HCGDNN-GRU2',
                ),
            ],
        ),
        # deepsig 201610A for gru1 gru2
        dict(
            type='AccuracyF1Plot',
            name='motivation_g1g2_deepsig_201610A.pdf',
            legend=legend,
            log_dir=log_dir,
            method=[
                # GRU1
                dict(
                    config='hcgdnn_abl_gru1_deepsig_iq_201610A',
                    name='HCGDNN-GRU1',
                ),
                # GRU2
                dict(
                    config='hcgdnn_abl_gru2_deepsig_iq_201610A',
                    name='HCGDNN-GRU2',
                ),
            ],
        ),

        # deepsig 201610A
        dict(
            type='AccuracyF1Plot',
            name='deepsig_201610A.pdf',
            legend=legend,
            log_dir=log_dir,
            method=[
                # deepsig 201610A compare with I/Q A/P
                dict(
                    config='hcgdnn_hcgnetv2_100_288_lr-0.00044-step-800_deepsig_iq_201610A',
                    name='HCGDNN',
                ),
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

                # deepsig 201601A compare with constellation and fb
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
    ],
    flops=dict(
        type='GetFlops',
        log_dir=log_dir,
        method=[
            # Fast MLDNN
            dict(
                config='hcgdnn_hcgnetv2_100_288_lr-0.00044-step-800_deepsig_iq_201610A',
                name='HCGDNN',
                input_shape=[(2, 1, 128), (2, 1, 128), (1, 128, 128)],
            ),
            dict(
                config='mldnn_mlnetv5_640_0.0004_0.5_deepsig_201610A',
                name='MLDNN',
                input_shape=[(1, 2, 128), (1, 2, 128), (1, 128, 128)],
            ),
        ],
    )
)
