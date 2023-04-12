import os.path as osp
from csrr.common.fileio import load as IOLoad
import matplotlib.pyplot as plt
import numpy as np

data_root = '/home/citybuster/Data/SignalProcessing/SignalRecognition/ChangShuo/CSRR2023'
json_name = ['validation.json', 'test.json', 'train.json', 'train_and_validation.json']


bws = []
for json in json_name:
    json_path = osp.join(data_root, json)
    annos = IOLoad(json_path)
    for anno in annos['annotations']:
        bws.extend(anno['bandwidth'])


bws = np.array(bws)
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