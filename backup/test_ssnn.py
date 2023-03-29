import os

import numpy as np
import scipy.io as sio

file_prefixes = ['val']
mod_labels = []
band_labels = []
snrs = [i for i in range(16, 22, 2)]
num_snr = len(snrs)
snr_range = [0]
num_mod = 5

for snr in snrs:
    for file_prefix in file_prefixes:
        file_path = os.path.join('/home/citybuster/Data/SignalProcessing/ModulationClassification/CSSS',
                                 f'{file_prefix}_{snr:02d}.mat')
        data = sio.loadmat(file_path)
        mod_label = data[file_prefix][0, 0][2].astype(np.int64)
        band_label = data[file_prefix][0, 0][3].astype(np.int64)
        mod_labels.append(mod_label)
        band_labels.append(band_label)

mod_labels = np.concatenate(mod_labels, axis=0)
band_labels = np.concatenate(band_labels, axis=0)

bands = []
mods = []
gt = np.zeros((mod_labels.shape[0], 25), dtype=np.float32)
for idx in range(mod_labels.shape[0]):
    band = np.zeros(5, dtype=np.float32)
    mod = np.zeros((5, 5), dtype=np.float32)
    for i in range(2):
        band[band_labels[idx, i]] = 1
        mod[band_labels[idx, i], mod_labels[idx, i]] = 1
        col_index = band_labels[idx, i] * 5 + mod_labels[idx, i]
        gt[idx, col_index] = 1.0
    bands.append(band)
    mods.append(mod)


def _eval(res_band, res_mod):
    y_ = np.zeros((1, 25), dtype=np.float32)
    sorted_anchor_index = np.argsort(res_band)
    for anchor_index in sorted_anchor_index[-2:]:
        y_[0, anchor_index * 5 + np.argmax(res_mod[anchor_index, :])] = 1.0

    return y_


y_ = []
for a, b in zip(bands, mods):
    item = _eval(a, b)
    y_.append(item)

y_ = np.concatenate(y_, axis=0)
dy = gt - y_
dy = np.sum(np.abs(dy), axis=1)
print(dy)
