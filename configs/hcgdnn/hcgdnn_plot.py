_base_ = [
    '../_base_/plot/amc.py',
]


log_dir = '/home/citybuster/Data/SignalProcessing/Workdir_Old'
legend = {
    'MLDNN': 0,
    'CLDNN': 1,
    'CNN4': 2,
    'DensCNN': 3,
    'ResCNN': 4,
    'CGDNN2': 5,
    'CLDNN2': 6,
    'DSCLDNN': 7,
    'VGGNet': 8,
    'AlexNet': 9,
    'ResNet': 10,
    'GoogleNet': 11,
    'SVM': 12,
    'DT': 13,
    'HCGDNN': 14,
    'HCGDNN-0.3': 15,
    'HCGDNN-0.09': 16,
    'HCGDNN-0.027': 17,
    'V1': 18,
    'V2': 19,
    'V3': 20,
    'V4': 21,
    'V4-0.3': 22,
    'V4-0.09': 23,
    'V4-0.027': 24,
    'V5': 25,
    'V5-0.3': 26,
    'V5-0.09': 27,
    'V5-0.027': 28,
    'V6': 29,
    'V6-0.3': 30,
    'V6-0.09': 31,
    'V6-0.027': 32,
    'HCGDNN-L1': 0,
    'HCGDNN-L2': 1,
    'HCGDNN-L3': 2,
    'HCGDNN-L': 3,
}

scatter = [
    'FM',
    'AM-SSB-SC',
    'AM-DSB-SC',
    'AM-SSB-WC',
    'AM-DSB-WC',
    'AM-DSB',
    'AM-SSB',
    'WBFM',
    '8PSK',
    'BPSK',
    'CPFSK',
    'GFSK',
    '4PAM',
    '16QAM',
    '64QAM',
    'QPSK',
    '32PSK',
    '16ASK',
    '32QAM',
    'GMSK',
    'OQPSK',
    '4ASK',
    '16PSK',
    '64APSK',
    '128QAM',
    '64QAM',
    '256QAM',
    'OOK',
]

plot = dict(
    type='CommonPlot',

    config='hcgdnn_abl_cg1g2_no_share_deepsig_iq_201610A',



    # Set the configs about confusion maps
    confusion_map=[
        dict(
            type='ConfusionMap',
        
            name='confusion_map_201610A.pdf',
            method=dict(
                config='hcgdnn_abl_cg1g2_no_share_deepsig_iq_201610A',
                name='HCGDNN',
            ),
        ),
    ],

    # Set the configs about loss accuracy
    train_test_curve=[
        dict(
            type='LossAccuracyPlot',
        
        
            name='loss_accuracy_201610A.pdf',
            method=dict(
                config='hcgdnn_abl_cg1g2_no_share_deepsig_iq_201610A',
                name='HCGDNN',
                train_metrics=['cnn_loss', 'gru1_loss', 'gru2_loss', 'loss'],
                test_metrics=['cnn_pre/snr_mean_all', 'gru1_pre/snr_mean_all',
                              'gru2_pre/snr_mean_all', 'final/snr_mean_all'],
                legend_suffix=['L1', 'L2', 'L3', 'L'],
            ),
        ),
    ],

    # Set the configs about snr accuracy and modulation F1 score
    snr_modulation=[
        # Ablation Study by an incremental way
        dict(
            type='AccuracyF1Plot',
            name='ablation_data.pdf',
        
        
            extra_pre=['0.3', '0.09', '0.027'],
            method=[
                # CNN
                dict(
                    config='hcgdnn_abl_cnn_deepsig_iq_201610A',
                    name='V1',
                ),
                # GRU1
                dict(
                    config='hcgdnn_abl_gru1_deepsig_iq_201610A',
                    name='V2',
                ),
                # GRU2
                dict(
                    config='hcgdnn_abl_gru2_deepsig_iq_201610A',
                    name='V3',
                ),
                # CNN+GRU1+NO-Share
                dict(
                    config='hcgdnn_abl_cg1_no_share_deepsig_iq_201610A',
                    name='V4',
                ),
                # CNN+GRU2+NO-Share
                dict(
                    config='hcgdnn_abl_cg2_no_share_deepsig_iq_201610A',
                    name='V5',
                ),
                # GRU1+GRU2+NO-Share
                dict(
                    config='hcgdnn_abl_g1g2_no_share_deepsig_iq_201610A',
                    name='V6',
                ),
                # Final Version+NO-Share
                dict(
                    config='hcgdnn_abl_cg1g2_no_share_deepsig_iq_201610A',
                    name='HCGDNN',
                ),
            ],
        ),

        # deepsig 201610A for cnn gru1
        dict(
            type='AccuracyF1Plot',
            name='motivation_cg1_deepsig_201610A.pdf',
        
        
            method=[
                # CNN
                dict(
                    config='hcgdnn_abl_cnn_deepsig_iq_201610A',
                    name='V1',
                ),
                # GRU1
                dict(
                    config='hcgdnn_abl_gru1_deepsig_iq_201610A',
                    name='V2',
                ),
            ],
        ),
        # deepsig 201610A for cnn gru2
        dict(
            type='AccuracyF1Plot',
            name='motivation_cg2_deepsig_201610A.pdf',
        
        
            method=[
                # CNN
                dict(
                    config='hcgdnn_abl_cnn_deepsig_iq_201610A',
                    name='V1',
                ),
                # GRU2
                dict(
                    config='hcgdnn_abl_gru2_deepsig_iq_201610A',
                    name='V3',
                ),
            ],
        ),
        # deepsig 201610A for gru1 gru2
        dict(
            type='AccuracyF1Plot',
            name='motivation_g1g2_deepsig_201610A.pdf',
        
        
            method=[
                # GRU1
                dict(
                    config='hcgdnn_abl_gru1_deepsig_iq_201610A',
                    name='V2',
                ),
                # GRU2
                dict(
                    config='hcgdnn_abl_gru2_deepsig_iq_201610A',
                    name='V3',
                ),
            ],
        ),

        # deepsig 201610A
        dict(
            type='AccuracyF1Plot',
            name='deepsig_201610A.pdf',
        
        
            method=[
                # deepsig 201610A compare with I/Q A/P
                dict(
                    config='hcgdnn_abl_cg1g2_no_share_deepsig_iq_201610A',
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
                    name='ResCNN',
                ),
                dict(
                    config='cldnn_deepsig_iq_201610A',
                    name='CLDNN',
                ),
                dict(
                    config='cnn4_deepsig_iq_201610A',
                    name='CNN4',
                ),
                dict(
                    config='denscnn_deepsig_iq_201610A',
                    name='DensCNN',
                ),

                # deepsig 201601A compare with constellation and fb
                dict(
                    config='mldnn_alexnetco_640_0.0004_0.5_deepsig_201610A',
                    name='AlexNet',
                ),
                dict(
                    config='mldnn_googlenetco_640_0.0004_0.5_deepsig_201610A',
                    name='GoogleNet',
                ),
                dict(
                    config='mldnn_resnetco_640_0.0004_0.5_deepsig_201610A',
                    name='ResNet',
                ),
                dict(
                    config='mldnn_vggnetco_640_0.0004_0.5_deepsig_201610A',
                    name='VGGNet',
                ),
                dict(
                    config='mldnn_cul_dt_feature_based',
                    name='SVM',
                ),
                dict(
                    config='mldnn_cul_svm_feature_based',
                    name='DT',
                ),
            ],
        ),
        # # deepsig 201801A
        # dict(
        #     type='AccuracyF1Plot',
        #     name='deepsig_201801A.pdf',
        # 
        # 
        #     method=[
        #         dict(
        #             config='hcgdnn_lr-0.00100_deepsig_iq_201801A',
        #             name='HCGDNN',
        #         ),
        #         dict(
        #             config='hcgdnn_lr-0.00150_deepsig_iq_201801A',
        #             name='HCGDNN',
        #         ),
        #     ],
        # ),
    ],
    flops=dict(
        type='GetFlops',
        method=[
            # HCGDNN
            dict(
                config='hcgdnn_abl_cg1g2_no_share_deepsig_iq_201610A',
                name='HCGDNN',
                input_shape=[(2, 1, 128), None, None],
            ),
            # CLDNN
            dict(
                config='cldnn_deepsig_iq_201610A',
                name='CLDNN',
                input_shape=[(1, 2, 128), None, None],
            ),
            dict(
                config='mldnn_mlnetv5_640_0.0004_0.5_deepsig_201610A',
                name='MLDNN',
                input_shape=[(1, 2, 128), (1, 2, 128), (1, 128, 128)],
            ),
        ],
    )
)
