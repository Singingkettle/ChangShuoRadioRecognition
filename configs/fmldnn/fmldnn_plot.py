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
    'Fast MLDNN-CE-CM-CN-EP': 43
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
    log_dir=log_dir,
    config='fmldnn_iq-ap-channel-deepsig-201610A',
    legend=legend,
    scatter=scatter,
    # Set the configs about confusion maps
    confusion_map=[
        dict(
            type='ConfusionMap',
            log_dir=log_dir,
            name='confusion_map_201610A.pdf',
            method=dict(
                config='fmldnn_iq-ap-channel-deepsig-201610A',
                name='Fast MLDNN',
            ),
        ),
    ],

    # Set the configs about snr accuracy and modulation F1 score
    snr_modulation=[
        # Ablation Study by an incremental way
        # dict(
        #     type='AccuracyF1Plot',
        #     name='ablation_data.pdf',
        #     legend=legend,
        #     log_dir=log_dir,
        #     method=[
        #         # CE
        #         dict(
        #             config='fmldnn_abl-cross-entropy_iq-ap-deepsig-201610A',
        #             name='Fast MLDNN-CE',
        #         ),
        #         # CE + CM
        #         dict(
        #             config='fmldnn_abl-cross-entropy_iq-ap-channel-deepsig-201610A',
        #             name='Fast MLDNN-CE-CM',
        #         ),
        #         # FL + CM
        #         dict(
        #             config='fmldnn_abl-focal_iq-ap-channel-deepsig-201610A',
        #             name='Fast MLDNN-FL-CM',
        #         ),
        #         # FL + CM + EP
        #         dict(
        #             config='fmldnn_fmlnetv46_abl-c-f-a_single-gpu_deepsig_201610A',
        #             name='Fast MLDNN-FL-CM-EP',
        #         ),
        #
        #         # CE + CM + CN
        #         dict(
        #             config='fmldnn_fmlnetv46_abl-c-s_single-gpu_deepsig_201610A',
        #             name='Fast MLDNN-CE-CM-CN',
        #         ),
        #         # FL + CM + CN
        #         dict(
        #             config='fmldnn_fmlnetv46_abl-c-f-s_single-gpu_deepsig_201610A',
        #             name='Fast MLDNN-FL-CM-CN',
        #         ),
        #         # FL + CM + CN + EP
        #         dict(
        #             config='fmldnn_fmlnetv46_abl-c-s-a_single-gpu_deepsig_201610A',
        #             name='Fast MLDNN-CE-CM-CN-EP',
        #         ),
        #     ],
        # ),

        # deepsig 201610A
        # deepsig 201610A compare with I/Q A/P
        dict(
            type='AccuracyF1Plot',
            name='deepsig_201610A_iq_ap.pdf',
            legend=legend,
            log_dir=log_dir,
            method=[
                dict(
                    config='mldnn_iq-ap-deepsig-201610A',
                    name='MLDNN',
                ),
                dict(
                    config='dscldnn_iq-ap-deepsig-201610A',
                    name='DSCLDNN',
                ),
                dict(
                    config='rescnn_iq-deepsig-201610A',
                    name='ResCNN-IQ',
                ),
                dict(
                    config='cldnn_iq-deepsig-201610A',
                    name='CLDNN-IQ',
                ),
                dict(
                    config='cnn4_iq-deepsig-201610A',
                    name='CNN4-IQ',
                ),
                dict(
                    config='denscnn_iq-deepsig-201610A',
                    name='DensCNN-IQ',
                ),
            ],
        ),
        # deepsig 201601A compare with constellation and fb
        dict(
            type='AccuracyF1Plot',
            name='deepsig_201610A_co_fb.pdf',
            legend=legend,
            log_dir=log_dir,
            method=[
                dict(
                    config='mldnn_iq-ap-deepsig-201610A',
                    name='MLDNN',
                ),
                dict(
                    config='alexnet_co-deepsig-201610A',
                    name='AlexNet-CO',
                ),
                dict(
                    config='googlenet_co-deepsig-201610A',
                    name='GoogleNet-CO',
                ),
                dict(
                    config='resnet_co-deepsig-201610A',
                    name='ResNet-CO',
                ),
                dict(
                    config='vggnet_co-deepsig-201610A',
                    name='VGGNet-CO',
                ),
                dict(
                    config='svm_feature-based_cumulants-deepsig-201610A',
                    name='SVM-FB',
                ),
                dict(
                    config='decisiontree_feature-based_cumulants-deepsig-201610A',
                    name='DecisionTree-FB',
                ),
            ],
        ),
    ],
    # summary=dict(
    #     type='ModulationSummary',
    #     name='.md',
    #     log_dir=log_dir,
    #     dataset=[
    #         '201610A',
    #         # '201801A',
    #     ],
    # ),
    # vis_fea=[
    #     dict(
    #         type='VisFea',
    #         log_dir=log_dir,
    #         name='vis_fea_201610A.pdf',
    #         method=dict(
    #             config='fmldnn_fmlnetv46-no-sa-ia-channel-spatial-attention_famcauxhead-focal-0.004-800_batchsize-640_lr-0.00069-0.3-300-500_dp-0.5_single-gpu_deepsig_201610A',
    #             name='Fast MLDNN',
    #         ),
    #     ),
    #     dict(
    #         type='VisFea',
    #         log_dir=log_dir,
    #         name='ce_cm_cn_vis_fea_201610A.pdf',
    #         method=dict(
    #             config='fmldnn_fmlnetv46_abl-c-s_single-gpu_deepsig_201610A',
    #             name='Fast MLDNN-CE-CM-CN',
    #         ),
    #     ),
    #     dict(
    #         type='VisFea',
    #         log_dir=log_dir,
    #         name='fl_cm_cn_vis_fea_201610A.pdf',
    #         method=dict(
    #             config='fmldnn_fmlnetv46_abl-c-f-s_single-gpu_deepsig_201610A',
    #             name='Fast MLDNN-FL-CM-CN',
    #         ),
    #     )
    # ],

    # flops=dict(
    #     type='GetFlops',
    #     log_dir=log_dir,
    #     method=[
    #         # CE
    #         dict(
    #             config='fmldnn_fmlnetv46_abl-h_single-gpu_deepsig_201610A',
    #             name='Fast MLDNN-CE',
    #             input_shape=[(1, 2, 128), (1, 2, 128), (1, 128, 128)],
    #         ),
    #         # CE + CM
    #         dict(
    #             config='fmldnn_fmlnetv46_abl-c_single-gpu_deepsig_201610A',
    #             name='Fast MLDNN-CE-CM',
    #             input_shape=[(2, 1, 128), (2, 1, 128), (1, 128, 128)],
    #         ),
    #         # FL + CM
    #         dict(
    #             config='fmldnn_fmlnetv46_abl-c-f_single-gpu_deepsig_201610A',
    #             name='Fast MLDNN-FL-CM',
    #             input_shape=[(2, 1, 128), (2, 1, 128), (1, 128, 128)],
    #         ),
    #         # FL + CM + EP
    #         dict(
    #             config='fmldnn_fmlnetv46_abl-c-f-a_single-gpu_deepsig_201610A',
    #             name='Fast MLDNN-FL-CM-EP',
    #             input_shape=[(2, 1, 128), (2, 1, 128), (1, 128, 128)],
    #         ),
    #
    #         # CE + CM + CN
    #         dict(
    #             config='fmldnn_fmlnetv46_abl-c-s_single-gpu_deepsig_201610A',
    #             name='Fast MLDNN-CE-CM-CN',
    #             input_shape=[(2, 1, 128), (2, 1, 128), (1, 128, 128)],
    #         ),
    #         # FL + CM + CN
    #         dict(
    #             config='fmldnn_fmlnetv46_abl-c-f-s_single-gpu_deepsig_201610A',
    #             name='Fast MLDNN-FL-CM-CN',
    #             input_shape=[(2, 1, 128), (2, 1, 128), (1, 128, 128)],
    #         ),
    #         # FL + CM + CN + EP
    #         dict(
    #             config='fmldnn_fmlnetv46_abl-c-s-a_single-gpu_deepsig_201610A',
    #             name='Fast MLDNN-CE-CM-CN-EP',
    #             input_shape=[(2, 1, 128), (2, 1, 128), (1, 128, 128)],
    #         ),
    #         # Fast MLDNN
    #         dict(
    #             config='fmldnn_fmlnetv46-no-sa-ia-channel-spatial-attention_famcauxhead-focal-0.004-800_batchsize-640_lr-0.00069-0.3-300-500_dp-0.5_single-gpu_deepsig_201610A',
    #             name='Fast MLDNN',
    #             input_shape=[(2, 1, 128), (2, 1, 128), (1, 128, 128)],
    #         ),
    #         dict(
    #             config='mldnn_mlnetv5_640_0.0004_0.5_deepsig_201610A',
    #             name='MLDNN',
    #             input_shape=[(1, 2, 128), (1, 2, 128), (1, 128, 128)],
    #         ),
    #     ]
    # )
)
