import numpy as np
import h5py
import copy
import os.path as osp
import pickle
from tqdm import tqdm


def shift_f(f, c):
    n = np.arange(0, 1024, 1)
    a = f * (n / 30 + c / 1200) * 2 * np.pi
    i = np.cos(a)
    q = -np.sin(a)
    iq = np.vstack((i, q))

    return iq


fs = np.arange(575, -600, -50)
cs = np.array([16, 11, 1, 0, 27, 24, 37, 31])
move = []
for f_i in range(24):
    item = np.zeros((16, 1024), dtype=np.float64)
    for c_i in range(8):
        iq = shift_f(fs[f_i], cs[c_i])
        item[c_i * 2:(c_i + 1) * 2, :] = iq
    move.append(copy.deepcopy(item))

file_name = 'data_2_test.h5'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/GBSense/2022/Advanced'
data = h5py.File(osp.join(data_root, file_name))
X = data['X'][:, :, :]
Y = data['Y'][:, :]

channel_label = []

c_num = 0
for i in tqdm(range(X.shape[0])):
    for j in range(24):
        item = X[i, :, :]
        item = np.transpose(item) * move[j]
        item = item.astype(np.float32)
        np.save(osp.join(data_root, 'shift/channel', f'test_{c_num:07d}.npy'), item)
        c_num += 1
        if Y[i, j] > 0:
            channel_label.append(1)
        else:
            channel_label.append(0)

pickle.dump(channel_label, open(osp.join(data_root, 'test_channel_label.pkl'), 'wb'), protocol=4)


mod_label = []
m_num = 0
for i in tqdm(range(X.shape[0])):
    for j in range(24):
        item = X[i, :, :]
        item = np.transpose(item) * move[j]
        item = item.astype(np.float32)
        if Y[i, j] > 0:
            np.save(osp.join(data_root, 'shift/mod', f'test_{m_num:07d}.npy'), item)
            m_num += 1
            mod_label.append(Y[i, j] - 1)

pickle.dump(mod_label, open(osp.join(data_root, 'test_mod_label.pkl'), 'wb'), protocol=4)


# file_name = 'data_2_train.h5'
# data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/GBSense/2022/Advanced'
# data = h5py.File(osp.join(data_root, file_name))
# X = data['X'][:, :, :]
# Y = data['Y'][:, :]

# channel_label = []

# c_num = 0
# for i in tqdm(range(X.shape[0])):
#     for j in range(24):
#         item = X[i, :, :]
#         item = np.transpose(item) * move[j]
#         item = item.astype(np.float32)
#         np.format(osp.join(data_root, 'shift/channel', f'train_{c_num:07d}.npy'), item)
#         c_num += 1
#         if Y[i, j] > 0:
#             channel_label.append(1)
#         else:
#             channel_label.append(0)

# pickle.dump(channel_label, open(osp.join(data_root, 'train_channel_label.pkl'), 'wb'), protocol=4)


# mod_label = []
# m_num = 0
# for i in tqdm(range(X.shape[0])):
#     for j in range(24):
#         item = X[i, :, :]
#         item = np.transpose(item) * move[j]
#         item = item.astype(np.float32)
#         if Y[i, j] > 0:
#             np.format(osp.join(data_root, 'shift/mod', f'train_{m_num:07d}.npy'), item)
#             m_num += 1
#             mod_label.append(Y[i, j] - 1)

# pickle.dump(mod_label, open(osp.join(data_root, 'train_mod_label.pkl'), 'wb'), protocol=4)
