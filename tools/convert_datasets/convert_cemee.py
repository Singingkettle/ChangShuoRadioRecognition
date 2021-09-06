#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Project: wtisignalprocessing
File: convert_cemee.py
Author: Citybuster
Time: 2021/8/21 12:06
Email: chagshuo@bupt.edu.cn
"""
import os
import os.path as osp

import h5py
import numpy as np


def load_train_mat(mat_file):
    """Load annotation from annotation file."""
    with h5py.File(mat_file, 'r') as f:
        X = np.array(f['X'], dtype=np.float32)
        X = np.transpose(X, (2, 1, 0))
        Y = np.array(f['Y'], dtype=np.int)
        Y = list(Y[0, :])

    return X, Y


mat_file = '/home/citybuster/Data/SignalProcessing/SpecificEmitterIdentification/CEMEE/2021-07/Task_1_Train.mat'
X, Y = load_train_mat(mat_file)

train_X = []
train_Y = []
val_X = []
val_Y = []
