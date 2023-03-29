import numpy as np

from csrr.common.fileio import load as IOLoad

iq1 = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201801A/cache/train_and_validation_iq.pkl'
iq1 = IOLoad(iq1)

iq2 = '/home/citybuster/cache_pkl/train_and_val-iq.pkl'
iq2 = IOLoad(iq2)
pass

for index in range(len(iq2)):
    iq11 = iq1['data'][index, :, :]
    iq21 = iq2[index]
    ab = iq11 - iq21
    ab = np.sum(ab[:])
    if ab != 0:
        print('fuck')
