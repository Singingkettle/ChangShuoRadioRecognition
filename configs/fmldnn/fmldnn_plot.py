_base_ = [
    '../_base_/plot/amc.py',
]

plot = dict(
    type='CommonPlot',
    name='FMLDNN',
    # Set the configs about confusion maps
    confusion_map=[
        dict(
            type='ConfusionMap',
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
        dict(
            type='AccuracyF1Plot',
            name='ablation_data.pdf',
            method=[
                dict(
                    config='fmldnn_abl-cross-entropy_iq-ap-deepsig-201610A',
                    name='Fast MLDNN-V1',
                ),
                dict(
                    config='fmldnn_abl-cross-entropy_iq-ap-channel-deepsig-201610A',
                    name='Fast MLDNN-V2',
                ),
                dict(
                    config='fmldnn_abl-focal_iq-ap-channel-deepsig-201610A',
                    name='Fast MLDNN-V3',
                ),
                dict(
                    config='fmldnn_abl-skip-focal_iq-ap-channel-deepsig-201610A',
                    name='Fast MLDNN-V4',
                ),
                dict(
                    config='fmldnn_iq-ap-channel-deepsig-201610A',
                    name='Fast MLDNN',
                ),
                dict(
                    config='fmldnn_abl-skip-cross-entropy_iq-ap-channel-deepsig-201610A',
                    name='Fast MLDNN-V5',
                ),
                dict(
                    config='fmldnn_abl-skip-cross-entropy-shrinkage_iq-ap-channel-deepsig-201610A',
                    name='Fast MLDNN-V6',
                ),
            ],
        ),

        # deepsig 201610A
        dict(
            type='AccuracyF1Plot',
            name='deepsig_201610A_iq_ap.pdf',
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
    summary=dict(
        type='ModulationSummary',
        name='.md',
        dataset=[
            '201610A',
            # '201801A',
        ],
    ),
    vis_fea=[
        dict(
            type='VisFea',
            name='fs_vis_fea_201610A.pdf',
            method=dict(
                config='fmldnn_iq-ap-channel-deepsig-201610A',
                name='Fast MLDNN',
            ),
        ),
        dict(
            type='VisFea',
            name='ce_vis_fea_201610A.pdf',
            method=dict(
                config='fmldnn_abl-cross-entropy_iq-ap-channel-deepsig-201610A',
                name='Fast MLDNN-V3',
            ),
        )
    ],

    flops=dict(
        type='GetFlops',
        method=[
            dict(
                config='fmldnn_abl-cross-entropy_iq-ap-deepsig-201610A',
                name='Fast MLDNN-V1',
                input_shape=[(1, 2, 128), (1, 2, 128), (1, 128, 128)],
            ),
            dict(
                config='fmldnn_abl-cross-entropy_iq-ap-channel-deepsig-201610A',
                name='Fast MLDNN-V2',
                input_shape=[(2, 1, 128), (2, 1, 128), (1, 128, 128)],
            ),
            dict(
                config='fmldnn_abl-focal_iq-ap-channel-deepsig-201610A',
                name='Fast MLDNN-V3',
                input_shape=[(2, 1, 128), (2, 1, 128), (1, 128, 128)],
            ),
            dict(
                config='fmldnn_abl-skip-focal_iq-ap-channel-deepsig-201610A',
                name='Fast MLDNN-V4',
                input_shape=[(2, 1, 128), (2, 1, 128), (1, 128, 128)],
            ),
            dict(
                config='fmldnn_iq-ap-channel-deepsig-201610A',
                name='Fast MLDNN',
                input_shape=[(2, 1, 128), (2, 1, 128), (1, 128, 128)],
            ),
            dict(
                config='fmldnn_abl-skip-cross-entropy_iq-ap-channel-deepsig-201610A',
                name='Fast MLDNN-V5',
                input_shape=[(2, 1, 128), (2, 1, 128), (1, 128, 128)],
            ),
            dict(
                config='fmldnn_abl-skip-cross-entropy-shrinkage_iq-ap-channel-deepsig-201610A',
                name='Fast MLDNN-V6',
                input_shape=[(2, 1, 128), (2, 1, 128), (1, 128, 128)],
            ),
        ]
    )
)
