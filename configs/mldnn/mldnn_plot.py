_base_ = [
    '../_base_/plot/amc.py',
]

plot = dict(
    type='CommonPlot',
    name='MLDNN',
    # Set the configs about confusion maps
    # confusion_map=[
    #     dict(
    #         type='ConfusionMap',
    #         name='confusion-map_mldnn-201610A.pdf',
    #         method=dict(
    #             config='mldnn_iq-ap-deepsig-201610A',
    #             name='MLDNN',
    #         ),
    #     ),
    # ],
    # Set the configs about snr accuracy and modulation F1 score
    snr_modulation=[
        # # Motivation Verification
        # dict(
        #     type='AccuracyF1Plot',
        #     name='motivation.pdf',
        #     method=[
        #         # CLDNN AP
        #         dict(
        #             config='cldnn_ap-deepsig-201610A',
        #             name='CLDNN-AP',
        #         ),
        #         # CLDNN IQ
        #         dict(
        #             config='cldnn_iq-deepsig-201610A',
        #             name='CLDNN-IQ',
        #         ),
        #         # CGDNN2 AP
        #         dict(
        #             config='cgdnn2_ap-deepsig-201610A',
        #             name='CGDNN2-AP',
        #         ),
        #         # CGDNN2 IQ
        #         dict(
        #             config='cgdnn2_iq-deepsig-201610A',
        #             name='CGDNN2-IQ',
        #         ),
        #         # CLDNN2 AP
        #         dict(
        #             config='cldnn2_ap-deepsig-201610A',
        #             name='CLDNN2-AP',
        #         ),
        #         # CLDNN2 IQ
        #         dict(
        #             config='cldnn2_iq-deepsig-201610A',
        #             name='CLDNN2-IQ',
        #         ),
        #     ]
        # ),
        #
        # # Ablation Study by an incremental way
        # dict(
        #     type='AccuracyF1Plot',
        #     name='ablation_data.pdf',
        #     method=[
        #         # V1, A/P
        #         dict(
        #             config='cnn3_ap-deepsig-201610A',
        #             name='CNN3-AP',  # MLDNN-V1
        #         ),
        #         # V2, I/Q
        #         dict(
        #             config='cnn3_iq-deepsig-201610A',
        #             name='CNN3-IQ',  # MLDNN-V2
        #         ),
        #         # V3
        #         dict(
        #             config='mldnn_abl-mtlh_iq-ap-deepsig-201610A',
        #             name='MLDNN-V3',
        #         ),
        #         # V4
        #         dict(
        #             config='mldnn_abl-bigru-mtlh_iq-ap-deepsig_201610A',
        #             name='MLDNN-V4',
        #         ),
        #         # V5
        #         dict(
        #             config='mldnn_abl-bigru-safn-mtlh_iq-ap-deepsig_201610A_abl',
        #             name='MLDNN-V5',
        #         ),
        #         # V6
        #         dict(
        #             config='mldnn_abl-gru-safn-gradient-truncation-mtlh_iq-ap-deepsig-201610A',
        #             name='MLDNN-V6',
        #         ),
        #         # V7
        #         dict(
        #             config='mldnn_abl-bigru-attention-gradient-truncation-mtlh_iq-ap-deepsig-201610A',
        #             name='MLDNN-V7',
        #         ),
        #         # V8
        #         dict(
        #             config='mldnn_abl-bigru-last-gradient-truncation-mtlh_iq-ap-deepsig-201610A',
        #             name='MLDNN-V8',
        #         ),
        #         # V9
        #         dict(
        #             config='mldnn_abl-bigru-add-gradient-truncation-mtlh_iq-ap-deepsig-201610A',
        #             name='MLDNN-V9',
        #         ),
        #         # MLDNN
        #         dict(
        #             config='mldnn_iq-ap-deepsig-201610A',
        #             name='MLDNN',
        #         ),
        #     ],
        # ),
        #
        # # deepsig 201610A compare with I/Q A/P
        # dict(
        #     type='AccuracyF1Plot',
        #     name='deepsig_201610A_iq_ap.pdf',
        #     method=[
        #         dict(
        #             config='mldnn_iq-ap-deepsig-201610A',
        #             name='MLDNN',
        #         ),
        #         dict(
        #             config='dscldnn_iq-ap-deepsig-201610A',
        #             name='DSCLDNN',
        #         ),
        #         dict(
        #             config='rescnn_iq-deepsig-201610A',
        #             name='ResCNN-IQ',
        #         ),
        #         dict(
        #             config='cldnn_iq-deepsig-201610A',
        #             name='CLDNN-IQ',
        #         ),
        #         dict(
        #             config='cnn4_iq-deepsig-201610A',
        #             name='CNN4-IQ',
        #         ),
        #         dict(
        #             config='denscnn_iq-deepsig-201610A',
        #             name='DensCNN-IQ',
        #         ),
        #     ],
        # ),
        # # deepsig 201601A compare with constellation and fb
        # dict(
        #     type='AccuracyF1Plot',
        #     name='deepsig_201610A_co_fb.pdf',
        #     method=[
        #         dict(
        #             config='mldnn_iq-ap-deepsig-201610A',
        #             name='MLDNN',
        #         ),
        #         dict(
        #             config='alexnet_co-deepsig-201610A',
        #             name='AlexNet-CO',
        #         ),
        #         dict(
        #             config='googlenet_co-deepsig-201610A',
        #             name='GoogleNet-CO',
        #         ),
        #         dict(
        #             config='resnet_co-deepsig-201610A',
        #             name='ResNet-CO',
        #         ),
        #         dict(
        #             config='vggnet_co-deepsig-201610A',
        #             name='VGGNet-CO',
        #         ),
        #         dict(
        #             config='svm_feature-based_cumulants-deepsig-201610A',
        #             name='SVM-FB',
        #         ),
        #         dict(
        #             config='decisiontree_feature-based_cumulants-deepsig-201610A',
        #             name='DecisionTree-FB',
        #         ),
        #     ],
        # ),
        #
        # # deepsig 201801A compare with I/Q A/P
        # dict(
        #     type='AccuracyF1Plot',
        #     name='deepsig_201801A_iq_ap.pdf',
        #     method=[
        #         dict(
        #             config='mldnn_iq-ap-deepsig-201801A',
        #             name='MLDNN',
        #         ),
        #         dict(
        #             config='dscldnn_iq-ap-deepsig-201801A',
        #             name='DSCLDNN',
        #         ),
        #         dict(
        #             config='rescnn_iq-deepsig-201801A',
        #             name='ResCNN-IQ',
        #         ),
        #         dict(
        #             config='cldnn_iq-deepsig-201801A',
        #             name='CLDNN-IQ',
        #         ),
        #         dict(
        #             config='cnn4_iq-deepsig-201801A',
        #             name='CNN4-IQ',
        #         ),
        #         dict(
        #             config='denscnn_iq-deepsig-201801A',
        #             name='DensCNN-IQ',
        #         ),
        #     ],
        # ),
        #
        # # deepsig 201801A compare with constellation and fb
        # dict(
        #     type='AccuracyF1Plot',
        #     name='deepsig_201801A_co_fb.pdf',
        #     method=[
        #         dict(
        #             config='mldnn_iq-ap-deepsig-201801A',
        #             name='MLDNN',
        #         ),
        #         dict(
        #             config='alexnet_co-deepsig-201801A',
        #             name='AlexNet-CO',
        #         ),
        #         dict(
        #             config='googlenet_co-deepsig-201801A',
        #             name='GoogleNet-CO',
        #         ),
        #         dict(
        #             config='resnet_co-deepsig-201801A',
        #             name='ResNet-CO',
        #         ),
        #         dict(
        #             config='vggnet_co-deepsig-201801A',
        #             name='VGGNet-CO',
        #         ),
        #         dict(
        #             config='svm_feature-based_cumulants-deepsig-201801A',
        #             name='SVM-FB',
        #         ),
        #         dict(
        #             config='decisiontree_feature-based_cumulants-deepsig-201801A',
        #             name='DecisionTree-FB',
        #         ),
        #     ],
        # ),

        # deepsig 201801A SNR[-8,30] compare with I/Q A/P
        dict(
            type='AccuracyF1Plot',
            name='snr_-8-30_deepsig_201801A_iq_ap.pdf',
            method=[
                dict(
                    config='mldnn_iq-ap-snr-[-8,20]-deepsig-201801A',
                    name='MLDNN',
                ),
                dict(
                    config='dscldnn_iq-ap-snr-[-8,20]-deepsig-201801A',
                    name='DSCLDNN',
                ),
                dict(
                    config='rescnn_iq-snr-[-8,20]-deepsig-201801A',
                    name='ResCNN-IQ',
                ),
                dict(
                    config='cldnn_iq-snr-[-8,20]-deepsig-201801A',
                    name='CLDNN-IQ',
                ),
                dict(
                    config='cnn4_iq-snr-[-8,20]-deepsig-201801A',
                    name='CNN4-IQ',
                ),
                dict(
                    config='denscnn_iq-snr-[-8,20]-deepsig-201801A',
                    name='DensCNN-IQ',
                ),
            ],
        ),

        # deepsig 201801A SNR[-8,30] compare with constellation and fb
        dict(
            type='AccuracyF1Plot',
            name='snr_-8-30_deepsig_201801A_co_fb.pdf',
            method=[
                dict(
                    config='mldnn_iq-ap-snr-[-8,20]-deepsig-201801A',
                    name='MLDNN',
                ),
                dict(
                    config='alexnet_co-snr-[-8,20]-deepsig-201801A',
                    name='AlexNet-CO',
                ),
                dict(
                    config='googlenet_co-snr-[-8,20]-deepsig-201801A',
                    name='GoogleNet-CO',
                ),
                dict(
                    config='resnet_co-snr-[-8,20]-deepsig-201801A',
                    name='ResNet-CO',
                ),
                dict(
                    config='vggnet_co-snr-[-8,20]-deepsig-201801A',
                    name='VGGNet-CO',
                ),
                dict(
                    config='svm_feature-based_cumulants-snr-[-8,20]-deepsig-201801A',
                    name='SVM-FB',
                ),
                dict(
                    config='decisiontree_feature-based_cumulants-snr-[-8,20]-deepsig-201801A',
                    name='DecisionTree-FB',
                ),
            ],
        ),
    ],
)
