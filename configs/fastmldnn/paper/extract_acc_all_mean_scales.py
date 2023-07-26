import os

import numpy as np
import pandas as pd

from csrr.common.utils.config import get_total_epoch, load_json_log
from csrr.common.utils.path import glob


def get_acc(json_path):
    final_metric = 'ACC'
    total_epoch = get_total_epoch(json_path.replace('.json', ''))
    merge_log = {i + 1: 0 for i in range(total_epoch)}
    log_dict = load_json_log(json_path)
    epochs = list(log_dict.keys())
    for epoch in epochs:
        if log_dict[epoch]['mode'][-1] == 'val':
            merge_log[epoch] = log_dict[epoch][final_metric][0]
    best_epoch = max(merge_log, key=merge_log.get)
    return merge_log[best_epoch]


accs = []
work_dir = '/home/citybuster/Data/WirelessRadio/work_dir'


x = []
y = []

for scale in np.arange(0.03, 1, 0.03):
    config = f'fastmldnn_abl-merge-mean-{scale:.3f}_iq-ap-channel-deepsig201610A'
    json_path = glob(os.path.join(work_dir, config), 'json')[0]
    acc = get_acc(json_path)
    x.append(scale)
    y.append(acc)

for scale in np.arange(5, 240, 4):
    config = f'fastmldnn_abl-merge-mean-{scale:03d}_iq-ap-channel-deepsig201610A'
    json_paths = glob(os.path.join(work_dir, config), 'json')[0]
    acc = get_acc(json_paths)
    x.append(scale)
    y.append(acc)

df = pd.DataFrame({'acc': y, 'scale': x})
df.to_csv("final.csv", index=False)