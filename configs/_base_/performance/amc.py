res_dir = '/home/citybuster/Data/SignalProcessing/Workdir'

legend = {
    'DecisionTree-FB': 0,
    'SVM-FB': 1,

    'AlexNet': 2,
    'GoogleNet': 3,
    'VGGNet': 4,
    'ResNet': 5,

    'CNN2': 6,
    'CNN3': 7,
    'CNN3_ap': 8,
    'CNN4': 9,
    'DensCNN': 10,
    'ResCNN': 11,

    'CGDNN2': 12,
    'CGDNN2_ap': 13,
    'CLDNN': 14,
    'CLDNN_ap': 15,
    'CLDNN2': 16,
    'CLDNN2_ap': 17,
    'DSCLDNN': 18,

    'MLDNN': 19,
    'MLDNN_V3': 20,
    'MLDNN_V4': 21,
    'MLDNN_V5': 22,
    'MLDNN_V6': 23,
    'MLDNN_V7': 24,
    'MLDNN_V8': 25,
    'MLDNN_V9': 26,

    'FastMLDNN': 27,
    'FastMLDNN_V1': 28,
    'FastMLDNN_V2': 29,
    'FastMLDNN_V3': 30,
    'FastMLDNN_V4': 31,
    'FastMLDNN_V5': 32,
    'FastMLDNN_V6': 33,

    'HCGDNN': 34,
    'HCGDNN_V1': 35,
    'HCGDNN_V2': 36,
    'HCGDNN_V3': 37,
    'HCGDNN_V4': 38,
    'HCGDNN_V5': 39,
    'HCGDNN_V6': 40,
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

publish = dict(
    deepsig201610A=dict(
        DecisionTree_FB='decisiontree_feature-based_cumulants-deepsig201610A',
        SVM_FB='svm_feature-based_cumulants-deepsig201610A',

        AlexNet='alexnet_co-deepsig201610A',
        GoogleNet='googlenet_co-deepsig201610A',
        VGGNet='vggnet_co-deepsig201610A',
        ResNet='resnet_co-deepsig201610A',

        CNN2='cnn2_iq-deepsig201610A',
        CNN3='cnn3_iq-deepsig201610A',
        CNN3_ap='cnn3_ap-deepsig201610A',
        CNN4='cnn4_iq-deepsig201610A',
        DensCNN='denscnn_iq-deepsig201610A',
        ResCNN='rescnn_iq-deepsig201610A',

        CGDNN2='cgdnn2_iq-deepsig201610A',
        CGDNN2_ap='cgdnn2_ap-deepsig201610A',
        CLDNN='cldnn_iq-deepsig201610A',
        CLDNN_ap='cldnn_ap-deepsig201610A',
        CLDNN2='cldnn2_iq-deepsig201610A',
        CLDNN2_ap='cldnn2_ap-deepsig201610A',

        DSCLDNN='dscldnn_iq-ap-deepsig201610A',

        MLDNN='mldnn_iq-ap-deepsig201610A',
        MLDNN_V3='mldnn_abl-mtlh_iq-ap-deepsig201610A',
        MLDNN_V4='mldnn_abl-bigru-mtlh_iq-ap-deepsig201610A',
        MLDNN_V5='mldnn_abl-bigru-safn-mtlh_iq-ap-deepsig201610A',
        MLDNN_V6='mldnn_abl-gru-safn-gradient-truncation-mtlh_iq-ap-deepsig201610A',
        MLDNN_V7='mldnn_abl-bigru-attention-gradient-truncation-mtlh_iq-ap-deepsig201610A',
        MLDNN_V8='mldnn_abl-bigru-last-gradient-truncation-mtlh_iq-ap-deepsig201610A',
        MLDNN_V9='mldnn_abl-bigru-add-gradient-truncation-mtlh_iq-ap-deepsig201610A',

        FastMLDNN='fmldnn_iq-ap-channel-deepsig201610A',
        FastMLDNN_V1='fmldnn_abl-cross-entropy_iq-ap-deepsig201610A',
        FastMLDNN_V2='fmldnn_abl-cross-entropy_iq-ap-channel-deepsig201610A',
        FastMLDNN_V3='fmldnn_abl-focal_iq-ap-channel-deepsig201610A',
        FastMLDNN_V4='fmldnn_abl-skip-focal_iq-ap-channel-deepsig201610A',
        FastMLDNN_V5='fmldnn_abl-skip-cross-entropy_iq-ap-channel-deepsig201610A',
        FastMLDNN_V6='fmldnn_abl-skip-cross-entropy-shrinkage_iq-ap-channel-deepsig201610A',

        HCGDNN='hcgdnn_iq-channel-deepsig201610A',
        HCGDNN_V1='hcgdnn_abl-cnn_iq-channel-deepsig201610A',
        HCGDNN_V2='hcgdnn_abl-bigru1_iq-channel-deepsig201610A',
        HCGDNN_V3='hcgdnn_abl-bigru2_iq-channel-deepsig201610A',
        HCGDNN_V4='hcgdnn_abl-cnn-bigru1_iq-channel-deepsig201610A',
        HCGDNN_V5='hcgdnn_abl-cnn-bigru2_iq-channel-deepsig201610A',
        HCGDNN_V6='hcgdnn_abl-bigru1-bigru2_iq-channel-deepsig201610A',
    ),
    deepsig201801A=dict(
        DecisionTree_FB='decisiontree_feature-based_cumulants-deepsig201801A',
        SVM_FB='svm_feature-based_cumulants-deepsig201801A',

        AlexNet='alexnet_co-deepsig201801A',
        GoogleNet='googlenet_co-deepsig201801A',
        VGGNet='vggnet_co-deepsig201801A',
        ResNet='resnet_co-deepsig201801A',

        CNN2='cnn2_iq-deepsig201801A',
        CNN3='cnn3_iq-deepsig201801A',
        CNN3_ap='cnn3_ap-deepsig201801A',
        CNN4='cnn4_iq-deepsig201801A',
        DensCNN='denscnn_iq-deepsig201801A',
        ResCNN='rescnn_iq-deepsig201801A',

        CGDNN2='cgdnn2_iq-deepsig201801A',
        CGDNN2_ap='cgdnn2_ap-deepsig201801A',
        CLDNN='cldnn_iq-deepsig201801A',
        CLDNN_ap='cldnn_ap-deepsig201801A',
        CLDNN2='cldnn2_iq-deepsig201801A',
        CLDNN2_ap='cldnn2_ap-deepsig201801A',

        DSCLDNN='dscldnn_iq-ap-deepsig201801A',

        MLDNN='mldnn_iq-ap-deepsig201801A',

        FastMLDNN='fmldnn_iq-ap-channel-deepsig201801A',
    ),
)
