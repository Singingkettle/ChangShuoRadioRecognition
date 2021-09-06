#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Project: wtisignalprocessing
File: compress.py
Author: Citybuster
Time: 2021/8/4 19:53
Email: chagshuo@bupt.edu.cn
"""
import glob
import os
import pickle
import zlib

import numpy as np
import tqdm

data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201801A/cache_pkl'
save_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201801A/zmq_cache_pkl_by_byte_compress'
os.chdir(data_root)
file_list = sorted(glob.glob("*.pkl"))
for file in file_list:
    data = pickle.load(open(file, 'rb'))
    for i, item in tqdm.tqdm(enumerate(data)):
        item = item.astype(np.float32)
        item = zlib.compress(item.tostring())
        data[i] = item
    pickle.dump(data, open(save_root + '/' + file, 'wb'))
