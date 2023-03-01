#
import matplotlib.pyplot as plt
import numpy as np

from csrr.common.fileio import load as IOLoad

data_path = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201610A/cache/validation_iq.pkl'

data = IOLoad(data_path)

t = np.arange(0, 128)
for item_index, item in enumerate(data['data']):
    fig, axs = plt.subplots()
    axs.set_title("Signal")
    axs.plot(t, item[0, :], 'g-', t, item[1, :], 'r-', )
    plt.show()
