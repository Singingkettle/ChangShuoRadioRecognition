#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Project: wtisignalprocessing
File: methods.py
Author: Citybuster
Time: 2021/9/28 22:06
Email: chagshuo@bupt.edu.cn
"""
import copy

import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint


def get_merge_weight_by_search(num_pre, search_step):
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


def get_merge_weight_by_optimization(x, t):
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
    res = minimize(min_obj, w0, method='trust-constr', jac=obj_der, hess=obj_hess,
                   constraints=[linear_constraint, ],
                   options={'verbose': 1}, bounds=bounds)
    best_w = res.x
    return best_w

## 下面的代码是用的CVX包的废弃代码
# def get_merge_weight(p, t):
#     p = p.astype(dtype=np.float64)
#     t = t.astype(dtype=np.float64)
#     n = p.shape[0]
#
#     G = -1 * spdiag(matrix(1.0, (n, 1)))
#     h = matrix(0.0, (n, 1))
#     A = matrix(1.0, (1, n))
#     b = matrix(1.0, (1, 1))
#
#     def F(x=None, z=None):
#         if x is None:
#             x0 = np.ones((n, 1), dtype=np.float64)/n
#             return 0, matrix(x0)
#         if min(x) < 0.0:
#             return None
#         w = np.array(x)
#         w = np.reshape(w.T, (n, 1, 1))
#
#         # forward pass
#         y0 = p * 100
#         y1 = y0 * w
#         y2 = np.sum(y1, axis=0)
#         y3 = np.exp(y2)
#         y4 = np.sum(y3, axis=1)[:, None]
#         y5 = y3 / y4
#         y6 = y5 - t
#         y7 = np.power(y6, 2)
#         f = np.sum(y7[:])
#
#         y8 = 2 * y6 * y5 * (1 - y5)
#         y9 = y8[None, :, :]
#         y10 = y9 * y0
#         Df = np.sum(np.sum(y10, axis=2), axis=1)
#         Df = np.reshape(Df, (1, -1))
#         Df = matrix(Df)
#         if z is None:
#             return f, Df
#         y11 = 2 * (-3 * np.power(y5, 2) + (2 + 2 * t) * y5 - t) * y5 * (1 - y5)
#         y12 = np.reshape(y11, (1, -1))
#         y13 = np.reshape(y0, (3, -1))
#         y14 = y13 * y12
#         H = y14 @ y13.T
#         H = matrix(H) * z[0]
#         return f, Df, H
#
#     sol = solvers.cp(F)
#     # sol = solvers.cp(F, G=G, h=h, A=A, b=b)
#
#     return np.array(sol['x'].T)

# def obj_fun(x, t, w, n):
#     w = np.reshape(w, (n, 1, 1))
#
#     # forward pass
#     y0 = x * 100
#     y1 = y0 * w
#     y2 = np.sum(y1, axis=0)
#     y3 = np.exp(y2)
#     y4 = np.sum(y3, axis=1)[:, None]
#     y5 = y3 / y4
#     y6 = y5 - t
#     y7 = np.power(y6, 2)
#     f = np.sum(y7[:]) / 1000
#
#     y8 = 2 * y6 * y5 * (1 - y5) / 1000 * t
#     y9 = y8[None, :, :]
#     y10 = y9 * y0
#     Df = np.sum(np.sum(y10, axis=2), axis=1)
#     Df = np.reshape(Df, (-1, 1))
#     return f, Df
