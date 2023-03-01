#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Project: ChangShuoRadioRecognition
File: flops.py
Author: Citybuster
Time: 2021/8/29 14:50
Email: chagshuo@bupt.edu.cn
"""

import os
from time import sleep

import torch

from ..builder import FLOPS
from ...common.utils import Config
from ...common.utils.flops_counter import get_model_complexity_info, flops_to_string, params_to_string, \
    inference_time_to_string
from ...models import build_method


@FLOPS.register_module()
class GetFlops:
    def __init__(self, method, log_dir):
        self.method = method
        self.log_dir = log_dir

    def _evaluate_model(self, cfg_path, name, input_shape):
        cfg = Config.fromfile(cfg_path)
        model = build_method(cfg.model)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()

        if hasattr(model, 'forward_dummy'):
            model.forward = model.forward_dummy
        else:
            raise NotImplementedError(
                'FLOPs counter is currently not currently supported with {}'.
                    format(model.__class__.__name__))
        # warm up
        for _ in range(1000):
            _, _, _ = get_model_complexity_info(model, input_shape,
                                                print_per_layer_state=False, as_strings=False)

        flops = 0
        params = 0
        mean_inference_time = 0
        for _ in range(1000):
            flops, params, inference_time = get_model_complexity_info(model, input_shape,
                                                                      print_per_layer_state=False, as_strings=False)
            mean_inference_time += inference_time
        inference_time = mean_inference_time / 1000

        flops = flops_to_string(flops)
        params = params_to_string(params)
        inference_time = inference_time_to_string(inference_time)

        split_line = '=' * 30
        print(f'{split_line}\nMethod: {name}\n'
              f'Input shape: {input_shape}\n'
              f'Flops: {flops}\nParams: {params}\nInference Time: {inference_time}\n{split_line}')
        print('!!!Please be cautious if you use the results in papers. '
              'You may need to check if all ops are supported and verify that the '
              'flops computation is correct.')

        del model
        return f'{split_line}\nMethod: {name}\n' \
               f'Input shape: {input_shape}\n' \
               f'Flops: {flops}\nParams: {params}\nInference Time: {inference_time}\n{split_line}'

    def plot(self, save_dir):
        save_path = os.path.join(save_dir, 'flops.txt')
        with open(save_path, 'w') as f:
            for method in self.method:
                cfg_path = os.path.join(self.log_dir, method['figure_configs'], method['figure_configs'] + '.py')
                flops_str = self._evaluate_model(cfg_path, method['name'], method['input_shape'])
                f.write(flops_str)
                torch.cuda.empty_cache()
                sleep(30)
