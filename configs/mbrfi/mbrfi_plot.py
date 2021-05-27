log_dir = '/home/citybuster/Data/SignalProcessing/Workdir'
legends = {
    'MBRFI-BPSK-resnet': 0,
    'MBRFI-QPSK-resnet': 1,
    'MBRFI-16QAM-resnet': 2,
    'MBRFI-32QAM-resnet': 3,
    'MBRFI-64QAM-resnet': 4,

    'MBRFI-BPSK-vggnet': 5,
    'MBRFI-QPSK-vggnet': 6,
    'MBRFI-16QAM-vggnet': 7,
    'MBRFI-32QAM-vggnet': 8,
    'MBRFI-64QAM-vggnet': 9,

    'MBRFI-BPSK-alexnet': 10,
    'MBRFI-QPSK-alexnet': 11,
    'MBRFI-16QAM-alexnet': 12,
    'MBRFI-32QAM-alexnet': 13,
    'MBRFI-64QAM-alexnet': 14,

    'MBRFI-16QAM-CON-Classic': 15,
    'MBRFI-16QAM-CON-EXP': 16,
    'MBRFI-16QAM-CON-Quadratic': 17,
    'MBRFI-16QAM-CON-Data': 18,

    'MBRFI-64QAM-CON-Classic': 19,
    'MBRFI-64QAM-CON-EXP': 20,
    'MBRFI-64QAM-CON-Quadratic': 21,
    'MBRFI-64QAM-CON-Data': 22,

    'MBRFI-64QAM-cobonet': 23,
}

plot = dict(
    type='MLDNNPlot',
    log_dir=log_dir,
    config='mbrfi',
    legends=legends,
    # Set the configs about snr accuracy and modulation F1 score
    snr_modulation=[
        dict(
            type='SNRModulationCurve',
            name='mbrfi_co.pdf',
            legends=legends,
            log_dir=log_dir,
            methods=[
                # dict(
                #     config='mbrfi_alexnetco_0.0001_16qam_chuan20210422',
                #     name='MBRFI-16QAM-alexnet',
                # ),
                # dict(
                #     config='mbrfi_alexnetco_0.0001_32qam_chuan20210422',
                #     name='MBRFI-32QAM-alexnet',
                # ),
                # dict(
                #     config='mbrfi_alexnetco_0.0001_64qam_chuan20210422',
                #     name='MBRFI-64QAM-alexnet',
                # ),
                # dict(
                #     config='mbrfi_alexnetco_0.0001_bpsk_chuan20210422',
                #     name='MBRFI-BPSK-alexnet',
                # ),
                # dict(
                #     config='mbrfi_alexnetco_0.0001_qpsk_chuan20210422',
                #     name='MBRFI-QPSK-alexnet',
                # ),
                # dict(
                #     config='mbrfi_resnetco_0.0001_16qam_chuan20210422',
                #     name='MBRFI-16QAM-resnet',
                # ),
                #
                # dict(
                #     config='mbrfi_resnetco_0.0001_64qam_chuan20210422',
                #     name='MBRFI-64QAM-resnet',
                # ),
                # dict(
                #     config='mbrfi_resnetco_0.0001_bpsk_chuan20210422',
                #     name='MBRFI-BPSK-resnet',
                # ),

                # dict(
                #     config='mbrfi_resnetco_0.0001_32qam_chuan20210422',
                #     name='MBRFI-32QAM-resnet',
                # ),
                # dict(
                #     config='mbrfi_resnetco_0.0001_qpsk_chuan20210422',
                #     name='MBRFI-QPSK-resnet',
                # ),
                # dict(
                #     config='mbrfi_vggnetco_0.0001_16qam_chuan20210422',
                #     name='MBRFI-16QAM-vggnet',
                # ),
                # dict(
                #     config='mbrfi_vggnetco_0.0001_32qam_chuan20210422',
                #     name='MBRFI-32QAM-vggnet',
                # ),
                # dict(
                #     config='mbrfi_vggnetco_0.0001_64qam_chuan20210422',
                #     name='MBRFI-64QAM-vggnet',
                # ),
                # dict(
                #     config='mbrfi_vggnetco_0.0001_bpsk_chuan20210422',
                #     name='MBRFI-BPSK-vggnet',
                # ),
                # dict(
                #     config='mbrfi_vggnetco_0.0001_qpsk_chuan20210422',
                #     name='MBRFI-QPSK-vggnet',
                # ),

                # dict(
                #     config='mbrfi_vggnetco_0.0005_con-data-classic-loss_16qam_chuan20210422',
                #     name='MBRFI-16QAM-CON-Classic',
                # ),
                # dict(
                #     config='mbrfi_vggnetco_0.0005_con-data-exp-loss_16qam_chuan20210422',
                #     name='MBRFI-16QAM-CON-EXP',
                # ),
                # dict(
                #     config='mbrfi_vggnetco_0.0005_con-data-quadratic-loss_16qam_chuan20210422',
                #     name='MBRFI-16QAM-CON-Quadratic',
                # ),
                # dict(
                #     config='mbrfi_vggnetco_0.0005_con-data_16qam_chuan20210422',
                #     name='MBRFI-16QAM-CON-Data',
                # ),

                # dict(
                #     config='mbrfi_vggnetco_0.0005_con-data-classic-loss_64qam_chuan20210422',
                #     name='MBRFI-64QAM-CON-Classic',
                # ),
                # dict(
                #     config='mbrfi_vggnetco_0.0005_con-data-exp-loss_64qam_chuan20210422',
                #     name='MBRFI-64QAM-CON-EXP',
                # ),
                # dict(
                #     config='mbrfi_vggnetco_0.0005_con-data-quadratic-loss_64qam_chuan20210422',
                #     name='MBRFI-64QAM-CON-Quadratic',
                # ),
                # dict(
                #     config='mbrfi_vggnetco_0.0005_con-data_64qam_chuan20210422',
                #     name='MBRFI-64QAM-CON-Data',
                # ),
                dict(
                    config='mbrfi_cobonet_0.00005_64qam_chuan20210422.py',
                    name='MBRFI-64QAM-cobonet',
                ),
            ],
        ),
    ],
)
