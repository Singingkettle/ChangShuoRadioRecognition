log_dir = '/home/citybuster/Data/SignalProcessing/Workdir'
legend = {
    'AlexNet-CO': 0,
    'CLDNN-IQ': 1,
    'CNN4-IQ': 2,
    'DecisionTree-FB': 3,
    'DensCNN-IQ': 4,
    'DSCLDNN': 5,
    'GoogleNet-CO': 6,
    'MLDNN': 7,
    'ResCNN-IQ': 8,
    'ResNet-CO': 9,
    'SVM-FB': 10,
    'VGGNet-CO': 11,
    'HCGDNN': 12,
    'FMLDNN': 13,
    'SEDNN': 14,
}

plot = dict(
    type='CommonPlot',
    log_dir=log_dir,
    config='sednn_iq-channel-snr-[-8,20]-deepsig-201801A',
    legend=legend,
    # Set the configs about confusion maps
    confusion_map=[
        dict(
            type='ConfusionMap',
            log_dir=log_dir,
            name='confusion-map_mldnn-201801A.pdf',
            method=dict(
                config='sednn_iq-ap-snr-[-8,20]-deepsig-201801A',
                name='SEDNN',
            ),
        ),
    ],
    # Set the configs about snr accuracy and modulation F1 score
    snr_modulation=[
        # deepsig 201801A compare with I/Q A/P
        dict(
            type='AccuracyF1Plot',
            name='deepsig_201801A_iq_ap.pdf',
            legend=legend,
            log_dir=log_dir,
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
                dict(
                    config='fmldnn_iq-ap-channel-snr-[-8,20]-deepsig-201801A',
                    name='FMLDNN',
                ),
                dict(
                    config='hcgdnn_iq-channel-snr-[-8,20]-deepsig-201801A',
                    name='HCGDNN',
                ),
            ],
        ),
        # deepsig 201801A compare with constellation and fb
        dict(
            type='AccuracyF1Plot',
            name='deepsig_201801A_co_fb.pdf',
            legend=legend,
            log_dir=log_dir,
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
