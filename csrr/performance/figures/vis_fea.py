#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Project: ChangShuoRadioRecognition
File: vis_fea.py
Author: Citybuster
Time: 2021/8/22 16:34
Email: chagshuo@bupt.edu.cn
"""
import os.path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ..builder import FIGURES


@FIGURES.register_module()
class VisFea:
    def __init__(self, dataset, legend_config=None, scatter_config=None):
        self.dataset = dataset
        self.legend_config = legend_config
        self.scatter_config = scatter_config

    def __call__(self, performances, save_dir):
        for dataset_name in self.dataset:
            for method_name in self.dataset[dataset_name]:
                img = performances[dataset_name][method_name]['Image/FeaDistribution']
                img = Image.fromarray(img)
                img.save(os.path.join(save_dir, f'{dataset_name}_{method_name}_FeaDistribution.png'))

