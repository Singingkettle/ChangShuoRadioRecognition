#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Project: wtisignalprocessing
File: transform.py
Author: Citybuster
Time: 2021/11/10 11:31
Email: chagshuo@bupt.edu.cn
"""

import numpy as np
import torch

from ..amc_data import _NORMMODES


class Transform(object):
    def __init__(self, dataset):
        self.iq = dataset.iq
        self.ap = dataset.ap
        self.co = dataset.co
        self.channel_mode = dataset.channel_mode
        self.filter = dataset.filter
        self.process_mode = dataset.process_mode
        self.data_aug = dataset.data_aug

        if self.co:
            # matrix window info
            self.height_range = [-1, 1]
            self.width_range = [-1, 1]
            self.filter_size = dataset.filter[0]
            self.filter_stride = dataset.filter[1]

    def _process(self, data, is_co=False):
        data = data.astype(np.float32)
        if is_co:
            data = np.expand_dims(data, axis=0)
        else:
            data = _NORMMODES[self.process_mode](data)
            num_path = 2
            if self.data_aug:
                aug_part = np.roll(data, -1) - data
                data = np.vstack([data, aug_part])
                num_path += 2
            if self.channel_mode:
                data = np.reshape(data, (num_path, 1, -1))
            else:
                data = np.reshape(data, (1, num_path, -1))
        data = np.expand_dims(data, axis=0)
        data = torch.from_numpy(data)
        data = data.to("cuda")
        return data

    def _axis_is(self, query_axis_x, query_axis_y):
        axis_x = query_axis_x // self.filter_stride
        axis_y = query_axis_y // self.filter_stride
        if axis_x * self.filter_stride + self.filter_size < query_axis_x:
            position = [None, None]
        elif axis_y * self.filter_stride + self.filter_size < query_axis_y:
            position = [None, None]
        else:
            position = [int(axis_x), int(axis_y)]
        return position

    def __call__(self, iq_data):

        input_data = dict()
        if self.iq:
            input_data['iqs'] = self._process(iq_data)
        if self.ap:
            amplitude = np.sqrt(np.sum(np.power(iq_data, 2), axis=0))
            phase = np.arctan(iq_data[0, :] / (iq_data[1, :] + np.finfo(np.float64).eps))
            ap_data = np.vstack((amplitude, phase))
            input_data['aps'] = self._process(ap_data)

        if self.co:
            matrix_width = int((self.width_range[1] - self.width_range[0] - self.filter_size) / self.filter_stride + 1)
            matrix_height = int(
                (self.height_range[1] - self.height_range[0] - self.filter_size) / self.filter_stride + 1)
            co_data = np.zeros((matrix_height, matrix_width))
            pos_list = map(self._axis_is, list(iq_data[0, :]), list(iq_data[1, :]))
            num_point = 0
            for pos in pos_list:
                if pos[0] is not None:
                    co_data[pos[0], pos[1]] += 1
                    num_point += 1
            input_data['cos'] = self._process(co_data)
        return input_data
