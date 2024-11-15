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

from ..builder import TABLES
from ...common.utils.flops_counter import get_model_complexity_info, flops_to_string, params_to_string, \
    inference_time_to_string
from ...models import build_method


@TABLES.register_module()
class Flops:
    def __init__(self, dataset, legend=None, scatter=None):
        self.dataset = dataset
        self.legend = legend
        self.scatter = scatter

    def __call__(self, performances, save_dir):

        content = ''
        for dataset_name in self.dataset:
            for method_name in self.dataset[dataset_name]:
                print(method_name)
                if method_name == 'MLDNN' or method_name == 'FastMLDNN_V1' or 'DSCLDNN':
                    input_shape = dict(iqs=(1, 2, 128), aps=(1, 2, 128))
                else:
                    input_shape = dict(iqs=(2, 128), aps=(2, 128))
                cfg = performances[dataset_name][method_name].cfg
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
                                                                              print_per_layer_state=False,
                                                                              as_strings=False)
                    mean_inference_time += inference_time
                inference_time = mean_inference_time / 1000

                flops = flops_to_string(flops)
                params = params_to_string(params)
                inference_time = inference_time_to_string(inference_time)

                split_line = '=' * 30
                del model
                content += f'{split_line}\nMethod: {method_name}\n ' \
                           f'Input shape: {input_shape}\n' \
                           f'Flops: {flops}\n ' \
                           f'Params: {params}\n' \
                           f'Inference Time: {inference_time}\n' \
                           f'{split_line}'
                torch.cuda.empty_cache()
                sleep(6)
            save_path = os.path.join(save_dir, f'flops.txt')
            with open(save_path, 'w') as f:
                f.write(content)
