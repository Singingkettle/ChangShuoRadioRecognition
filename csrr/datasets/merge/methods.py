#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Project: ChangShuoRadioRecognition
File: methods.py
Author: Citybuster
Time: 2021/9/28 22:06
Email: chagshuo@bupt.edu.cn
"""
import copy

import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize

from ..builder import MERGES


def get_merge_weight_by_grid_search(num_pre, search_step):
    def search(search_depth, cur_sum):
        if search_depth == 1:
            return [[cur_sum]]
        search_depth -= 1
        cur_val = 0
        cur_res = []
        while cur_val <= cur_sum:
            sub_res = search(search_depth, cur_sum - cur_val)
            for sample in sub_res:
                new_sample = copy.deepcopy(sample)
                new_sample.append(cur_val)
                cur_res.append(new_sample)
            cur_val += search_step

        return cur_res

    res = search(num_pre, 1)

    return res


def get_merge_weight_by_optimization(x, t, method):
    x = x.astype(dtype=np.float64)
    t = t.astype(dtype=np.float64)
    m = x.shape[0]
    n = x.shape[1]
    tau = 745 / (np.max(x[:]) + np.finfo(np.float64).eps)

    r1 = 2 * tau / n
    r2 = 2 * tau * tau / n

    def min_obj(w):
        w = np.reshape(w, (-1, 1, 1))
        y0 = x * w
        y1 = np.sum(y0, axis=0)
        y2 = y1 * tau
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
        y2 = y1 * tau
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
        y2 = y1 * tau
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

    w0 = np.zeros((m,), dtype=np.float64)
    w0[-1] = 1

    lb = [0, ] * m
    ub = [1, ] * m
    bounds = Bounds(lb, ub)

    A = [[1, ] * m, ]
    lb = [1, ]
    ub = [1, ]
    linear_constraint = LinearConstraint(A, lb, ub)
    res = minimize(min_obj, w0, method=method, jac=obj_der, hess=obj_hess,
                   constraints=[linear_constraint, ],
                   options={'verbose': 1}, bounds=bounds)
    best_w = res.x
    return best_w


@MERGES.register_module()
class GridSearch:
    def __init__(self, grid_step):
        self.grid_step = grid_step

    def __call__(self, mpps, gts):
        """

        Args:
            mpps (List[np.ndarry]):
            gts:
        Returns:

        """
        mpps = np.stack(mpps, axis=0)
        m, n, c = mpps.shape
        ws = get_merge_weight_by_grid_search(m, self.grid_step)
        ws = np.array(ws, dtype=np.float64)
        pps = np.reshape(mpps, (m, -1))
        pps = ws @ pps
        pps = np.reshape(pps, (-1, n, c))
        pps_ = np.argmax(pps, axis=2)

        max_acc = 0
        best_pps = pps[0, :, :]
        for w_index in range(ws.shape[0]):
            cur_acc = accuracy_score(gts, pps_[w_index, :])
            if cur_acc > max_acc:
                max_acc = cur_acc
                best_pps = pps[w_index, :, :]

        return best_pps


@MERGES.register_module()
class Optimization:
    def __init__(self, method='trust-constr'):
        self.method = method

    def __call__(self, mpps, gts):
        """

        Args:
            mpps (List[np.ndarry]):
            gts:
        Returns:

        """

        mpps = np.stack(mpps, axis=0)
        m, n, c = mpps.shape
        gts = label_binarize(gts, classes=[i for i in range(c)])

        pre_max_index = np.argmax(mpps, axis=2)
        gt_max_index = np.argmax(gts, axis=1)
        diff_index = pre_max_index - np.reshape(gt_max_index, (1, -1))
        no_zero_index = np.nonzero((np.sum(np.abs(diff_index), axis=0)))[0]

        bad_pre_matrix = mpps[:, no_zero_index[:], :]
        targets = gts[no_zero_index[:], :]

        w = get_merge_weight_by_optimization(bad_pre_matrix, targets, self.method)
        mpps = np.dot(w.T, np.reshape(mpps, (m, -1)))
        pps = np.reshape(mpps, (n, c))

        return pps
