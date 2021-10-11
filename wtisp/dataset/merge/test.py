#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Project: wtisignalprocessing
File: test.py
Author: Citybuster
Time: 2021/9/30 11:40
Email: chagshuo@bupt.edu.cn
"""

import numpy as np


a1 = np.array([[0.1, 0.9], [0.2, 0.8]])
a2 = np.array([[0.15, 0.85], [0.25, 0.75]])
a3 = np.array([[0.05, 0.95], [0.35, 0.65]])
x = np.concatenate([a1[None, :, :], a2[None, :, :], a3[None, :, :]], axis=0)
w1 = np.array([0.2, 0.2, 0.6])
w1 = np.reshape(w1, (-1, 1, 1))
w2 = np.array([0.200000005, 0.2, 0.6])
w2 = np.reshape(w2, (-1, 1, 1))

t = np.array([[0,1],[0,1]])

m = x.shape[0]
n = x.shape[1]

r1 = 1490 / n
r2 = 110050 / n

def min_obj(w):
    w = np.reshape(w, (-1, 1, 1))
    y0 = x * w
    y1 = np.sum(y0, axis=0)
    y2 = y1 * 745
    y2 = y2 - np.max(y2, axis=1)[:, None]
    y3 = np.exp(y2)
    y4 = np.sum(y3, axis=1)[:, None]
    y5 = y3 / y4

    y6 = (y5 - t)
    y7 = np.power(y6, 2)
    f = np.mean(y7[:])
    return f

def obj_der(w):
    w = np.reshape(w, (-1, 1, 1))
    y0 = x * w
    y1 = np.sum(y0, axis=0)
    y2 = y1 * 745
    y2 = y2 - np.max(y2, axis=1)[:, None]
    y3 = np.exp(y2)
    y4 = np.sum(y3, axis=1)[:, None]
    y5 = y3 / y4

    y6 = y5 * (y5 - t)[None, :, :]

    y7 = y5[None, :, :]
    y7 = y7 * x
    y7 = np.sum(y7, axis=2)[:, :, None]

    y8 = y6 * (x - y7)

    df = np.sum(np.sum(y8, axis=2), axis=1) * r1

    return df

def obj_hess(w):
    w = np.reshape(w, (-1, 1, 1))
    y0 = x * w
    y1 = np.sum(y0, axis=0)
    y2 = y1 * 745
    y2 = y2 - np.max(y2, axis=1)[:, None]
    y3 = np.exp(y2)
    y4 = np.sum(y3, axis=1)[:, None]
    y5 = y3 / y4
    y6 = y5[None, :, :]

    y7_ = y5 * (2 * y5 - t)
    y7 = y7_[None, :, :]
    y8_ = np.sum(y6 * x, axis=2)
    y8 = y8_[:, :, None]
    y9 = x - y8
    y10 = y7 * y9

    y11 = np.reshape(y9, (m, -1))
    y12 = np.reshape(y10, (m, -1))
    H1 = y11 @ y12.T

    y13 = y5 * (y5 - t)
    y13 = np.sum(y13, axis=1)
    y13 = np.reshape(y13, (1, -1))

    y14 = y8_ * y13
    y15 = y14 @ y8_.T

    y16 = y6 * x * y13[:, :, None]
    y17 = np.reshape(y16, (m, -1))
    y18 = np.reshape(x, (m, -1))

    H2 = y15 - y17 @ y18.T

    H = H1 + H2

    H = H * r2

    return H


df = (min_obj(w2)-min_obj(w2))/0.000000005
df_ = obj_der(w1)[0]
print(df - df_)