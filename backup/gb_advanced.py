import h5py
import numpy as np


file_paths = ['/home/citybuster/Data/SignalProcessing/ModulationClassification/GBSense/2022/Advanced/data_2_train.h5',
              '/home/citybuster/Data/SignalProcessing/ModulationClassification/GBSense/2022/Advanced/data_2_test.h5',]


a = 0
b = 0
c = 0
for file_path in file_paths:
    data = h5py.File(file_path)
    target = data['Y'][:, :]
    for item_index in range(target.shape[0]):
        Y = target[item_index, :] > 0
        a += 1
        if np.sum(Y) == 1:
            b += 1
            print(np.sum(Y))
            print(file_path)
            print(item_index)
            print('======================')

        if np.sum(Y) == 0:
            c += 1
            print(np.sum(Y))
            print(file_path)
            print(item_index)
            print('**************************')

print(a)
print(b)
print(c)


a = np.array([0, 1, 0, 0])
print(a)
a = np.nonzero(a)
print(a)