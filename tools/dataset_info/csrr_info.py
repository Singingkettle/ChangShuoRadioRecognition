import os.path as osp

import numpy as np
import json

json_name = ['test.json', 'train_and_validation.json']

bws = []
fs = []
nums = []
mods = []
for version in range(1, 42):
    data_root = f'D:/Data/ModulationClassification/Paper/data/ChangShuo/v{version:d}'
    for json_file in json_name:
        json_path = osp.join(data_root, json_file)
        with open(json_path, 'r') as f:
            annos = json.load(f)
        for anno in annos['annotations']:
            bws.extend(anno['bandwidth'])
            nums.append(len(anno['bandwidth']))
            fs.extend(anno['center_frequency'])
            mods.extend(anno['modulation'])

nums = np.array(nums)
bws = np.array(bws)

np.save('nums', nums)
np.save('bws', bws)
np.save('fs', fs)

ms = dict()
for mod in mods:
    if mod in ms:
        ms[mod] += 1
    else:
        ms[mod] = 1

print(ms)

plt.hist(nums)
plt.show()

bws = np.array(bws) / 150e3 * 1200
plt.hist(bws, bins = 8, density=True, cumulative = True,)
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