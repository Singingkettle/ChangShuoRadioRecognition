log_dir = '/home/citybuster/Data/SignalProcessing/Workdir'
legend = {
    'MLDNN': 0,
    'CLDNN': 1,
    'CLDNN-AP': 2,
    'CNN2': 3,
    'CNN2-AP': 4,
    'CNN3': 5,
    'CNN3-AP': 6,
    'CNN4': 7,
    'CNN4-AP': 8,
    'DensCNN': 9,
    'DensCNN-AP': 10,
    'ResCNN': 11,
    'ResCNN-AP': 12,
    'CGDNN2': 13,
    'CGDNN2-AP': 14,
    'CLDNN2': 15,
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
    'VGGNet': 29,
    'AlexNet': 30,
    'ResNet': 31,
    'GoogleNet': 32,
    'SVM': 33,
    'DecisionTree': 34,
    'MCBLDN': 35,
    'SLCBDN': 36,
    'SCBDN_R': 37,
    'LCBDN_R': 38,
    'SLCBDN_R': 39,
    'MCBLDN_R': 40,
    'MCBLDN_A': 41,
    'SCBDN_A': 42,
    'LCBDN_A': 43,
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
    config='cpcstn_deepsig_2018',
    legend=legend,
    scatter=scatter,
    # Set the configs about loss accuracy
    train_test_curve=[
        dict(
            type='LossAccuracyPlot',
            log_dir=log_dir,
            legend=legend,
            name='loss_accuracy_2018.pdf',
            method=dict(
                config='cpcstn_deepsig_2018',
                name='SLCBDN',
                train_metrics=['loss'],
                test_metrics=['common/snr_mean_all'],
            ),
        ),
    ],

    #  Set the configs about snr accuracy and modulation F1 score
    # snr_modulation=[
        #
        # dict(
        #     type='AccuracyF1Plot',
        #     name='slot.pdf',
        #     legend=legend,
        #     log_dir=log_dir,
        #     method=[
        #         dict(
        #             config='cpcstn_matslot_Con_128_Ds10_4000_Adam_v5_0.0002',
        #             name='SLCBDN',
        #         ),
        #         dict(
        #             config='con_128_ds10_4000_mcbldn_Adam',
        #             name='MCBLDN_A',
        #         ),
        #         dict(
        #             config='cpcstn_dscldnn_slot',
        #             name='DSCLDNN',
        #         ),
        #         dict(
        #             config='cpcstn_rescnn_iq_slot',
        #             name='ResCNN',
        #         ),
        #         dict(
        #             config='cpcstn_cldnn_iq_slot',
        #             name='CLDNN',
        #         ),
        #         dict(
        #             config='cpcstn_cnn2_iq_slot',
        #             name='CNN2',
        #         ),
        #         dict(
        #             config='cpcstn_denscnn_iq_slot',
        #             name='DensCNN',
        #         ),
        #         dict(
        #             config='cpcstn_alexnetco_slot',
        #             name='AlexNet',
        #         ),
        #         dict(
        #             config='cpcstn_googlenetco_slot',
        #             name='GoogleNet',
        #         ),
        #         dict(
        #             config='cpcstn_resnetco_slot',
        #             name='ResNet',
        #         ),
        #         dict(
        #             config='cpcstn_cul_dt_feature_based_slot',
        #             name='DecisionTree',
        #         ),
        #         dict(
        #             config='cpcstn_cul_svm_feature_based_slot',
        #             name='SVM',
        #         ),
        #     ],
        # ),
        #
        # # deepsig 201801A
        # dict(
        #     type='AccuracyF1Plot',
        #     name='deepsig_201801A.pdf',
        #     legend=legend,
        #     log_dir=log_dir,
        #     method=[
        #         dict(
        #             config='cpcstn_deepsig_2018',
        #             name='SLCBDN',
        #         ),
        #         dict(
        #             config='mcbldn_deepsig_2018',
        #             name='MCBLDN_A',
        #         ),
        #         dict(
        #             config='dscldnn_deepsig_201801A',
        #             name='DSCLDNN',
        #         ),
        #         dict(
        #             config='rescnn_deepsig_iq_201801A',
        #             name='ResCNN',
        #         ),
        #         dict(
        #             config='cldnn_deepsig_iq_201801A',
        #             name='CLDNN',
        #         ),
        #         dict(
        #             config='cnn2_deepsig_iq_201801A',
        #             name='CNN2',
        #         ),
        #         dict(
        #             config='denscnn_deepsig_iq_201801A',
        #             name='DensCNN',
        #         ),
        #         dict(
        #             config='mldnn_alexnetco_640_0.0004_0.5_deepsig_201801A',
        #             name='AlexNet',
        #         ),
        #         dict(
        #             config='mldnn_googlenetco_640_0.0004_0.5_deepsig_201801A',
        #             name='GoogleNet',
        #         ),
        #         dict(
        #             config='mldnn_resnetco_640_0.0004_0.5_deepsig_201801A',
        #             name='ResNet',
        #         ),
        #         dict(
        #             config='mldnn_cul_dt_feature_based_deepsig_201801A',
        #             name='DecisionTree',
        #         ),
        #         dict(
        #             config='mldnn_cul_svm_feature_based_deepsig_201801A',
        #             name='SVM',
        #         ),
        #     ],
        # ),

    #     dict(
    #         type='AccuracyF1Plot',
    #         name='ablation_Adam.pdf',
    #         legend=legend,
    #         log_dir=log_dir,
    #         method=[
    #             # MCBLDN
    #             dict(
    #                 config='con_128_ds10_4000_mcbldn',
    #                 name='MCBLDN',
    #             ),
    #             dict(
    #                 config='con_128_ds10_4000_mcbldn_Adam',
    #                 name='MCBLDN_A',
    #             ),
    #             # SCBDN
    #             dict(
    #                 config='cstn_matslot_Con_128_Ds10_4000_v5',
    #                 name='SCBDN_R',
    #             ),
    #             dict(
    #                 config='cstn_matslot_Con_128_Ds10_4000_Adam_v5_0.0002',
    #                 name='SCBDN_A',
    #             ),
    #             # LCBDN
    #             dict(
    #                 config='cpcnn_matslot_Con_128_Ds10_4000',
    #                 name='LCBDN_R',
    #             ),
    #             dict(
    #                 config='cpcnn_matslot_Con_128_Ds10_4000_Adam',
    #                 name='LCBDN_A',
    #             ),
    #             # SLCBDN
    #             dict(
    #                 config='cpcstn_matslot_Con_128_Ds10_4000_v5',
    #                 name='SLCBDN_R',
    #             ),
    #             dict(
    #                 config='cpcstn_matslot_Con_128_Ds10_4000_Adam_v5_0.0002',
    #                 name='SLCBDN',
    #             ),
    #         ],
    #     ),
    #
    # ],
    # flops=dict(
    #     type='GetFlops',
    #     log_dir=log_dir,
    #     method=[
    #         dict(
    #             config='cpcstn_deepsig_2018',
    #             name='SLCBDN',
    #             input_shape=[None, None, (8, 128, 128)],
    #         ),
    #         dict(
    #             config='mcbldn_deepsig_2018',
    #             name='MCBLDN',
    #             input_shape=[None, None, (8, 128, 128)],
    #         ),
    #     ],
    # )
)
