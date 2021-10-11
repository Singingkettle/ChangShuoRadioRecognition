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
}

plot = dict(
    type='MLDNNPlot',
    log_dir=log_dir,
    config='mldnn_mlnetv5_640_0.0004_0.5_deepsig_201801A',
    legend=legend,

    # Set the configs about snr accuracy and modulation F1 score
    snr_modulation=[
        # deepsig 201601A compare with constellation and fb
        dict(
            type='AccuracyF1Plot',
            name='deepsig_201801A_co_fb.pdf',
            legend=legend,
            log_dir=log_dir,
            method=[
                dict(
                    config='mldnn_alexnetco_640_0.0004_0.5_deepsig_201801A',
                    name='AlexNet-CO',
                ),
                dict(
                    config='mldnn_googlenetco_640_0.0004_0.5_deepsig_201801A',
                    name='GoogleNet-CO',
                ),
                dict(
                    config='mldnn_resnetco_640_0.0004_0.5_deepsig_201801A',
                    name='ResNet-CO',
                ),
                dict(
                    config='mldnn_vggnetco_640_0.0004_0.5_deepsig_201801A',
                    name='VGGNet-CO',
                ),
                dict(
                    config='mldnn_cul_dt_feature_based_deepsig_201801A',
                    name='SVM-FB',
                ),
                dict(
                    config='mldnn_cul_svm_feature_based_deepsig_201801A',
                    name='DecisionTree-FB',
                ),
            ],
        ),
    ],
    summary=dict(
        type='ModulationSummary',
        name='.md',
        log_dir=log_dir,
        dataset=[
            '201801A',
            '201801A',
        ],
    ),
)
