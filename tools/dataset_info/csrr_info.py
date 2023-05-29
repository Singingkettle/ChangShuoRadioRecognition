import os.path as osp

import matplotlib.pyplot as plt
import numpy as np

from csrr.common.fileio import load as IOLoad

data_root = './data/ChangShuo/v1'
json_name = ['validation.json', 'test.json', 'train.json', 'train_and_validation.json']

bws = []
nums = []
for version in range(1, 2):
    data_root = f'./data/ChangShuo/v{version:d}'
    for json in json_name:
        json_path = osp.join(data_root, json)
        annos = IOLoad(json_path)
        for anno in annos['annotations']:
            bws.extend(anno['bandwidth'])
            nums.append(len(anno['bandwidth']))

nums = np.array(nums)
plt.hist(nums)
plt.show()

bws = np.array(bws) / 150000 * 1200
plt.hist(bws)
plt.show()

bws = np.reshape(bws, [-1, 1])
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=0, init=np.reshape(np.array([1000, 1500, 3200, 60000]), [-1, 1])).fit(bws)
y = kmeans.labels_
print(kmeans.cluster_centers_)

import matplotlib.pyplot as plt

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
axs.scatter(bws, bws, c=y)
plt.show()