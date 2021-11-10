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
    'HCLDNN': 100,
}

plot = dict(
    type='CommonPlot',
    log_dir=log_dir,
    config='mct_201610A',
    legend=legend,
    # Set the configs about snr accuracy and modulation F1 score
    snr_modulation=[

        # Motivations
        dict(
            type='AccuracyF1Plot',
            name='cldnn_iq-ap.pdf',
            legend=legend,
            log_dir=log_dir,
            method=[
                dict(
                    config='mct_netv2-40-256_lr-0.0003_deepsig_201610A',
                    name='CLDNN-IQ',
                ),
                dict(
                    config='mct_netv2-40-256_lr-0.0004_deepsig_201610A',
                    name='CLDNN-IQ',
                ),
                dict(
                    config='mct_netv2-40-256_lr-0.0005_deepsig_201610A',
                    name='CLDNN-IQ',
                ),
                dict(
                    config='mct_netv2-40-256_lr-0.0006_deepsig_201610A',
                    name='CLDNN-IQ',
                ),
                dict(
                    config='mct_netv2-40-256_lr-0.0007_deepsig_201610A',
                    name='CLDNN-IQ',
                ),
                dict(
                    config='mct_netv2-40-256_lr-0.0008_deepsig_201610A',
                    name='CLDNN-IQ',
                ),
                dict(
                    config='mct_netv2-40-256_lr-0.0009_deepsig_201610A',
                    name='CLDNN-IQ',
                ),
                dict(
                    config='mct_netv2-40-256_lr-0.0010_deepsig_201610A',
                    name='CLDNN-IQ',
                ),
            ],
        ),
    ]
)
