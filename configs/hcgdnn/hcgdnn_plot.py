_base_ = [
    '../_base_/plot/amc.py',
]

plot = dict(
    type='CommonPlot',
    config='hcgdnn_iq-channel-deepsig-201610A',
    # Set the configs about confusion maps
    confusion_map=[
        dict(
            type='ConfusionMap',
            name='confusion-map_hcgdnn_deepsig-201610A.pdf',
            method=dict(
                config='hcgdnn_iq-channel-deepsig-201610A.py',
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
                config='hcgdnn_iq-channel-deepsig-201610A.py',
                name='HCGDNN',
                train_metrics=['loss_CNN', 'loss_GRU1', 'loss_GRU2', 'loss'],
                test_metrics=['CNN/snr_mean_all', 'GRU1/snr_mean_all',
                              'GRU2/snr_mean_all', 'Final/snr_mean_all'],
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
            method=[
                # CNN
                dict(
                    config='hcgdnn_abl-cnn_iq-channel-deepsig-201610A',
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
