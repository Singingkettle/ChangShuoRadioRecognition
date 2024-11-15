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
from .base import BaseDraw

@FIGURES.register_module()
class VisFea(BaseDraw):
    def __init__(self, dataset, legend=None, scatter=None):
        super().__init__(dataset)
        self.dataset = dataset
        self.legend = legend
        self.scatter = scatter

    def __call__(self, performances, save_dir):
        for dataset_name in self.dataset:
            for method_name in self.dataset[dataset_name]:
                if isinstance(method_name, str):
                    fea_infos = performances[dataset_name][method_name].FeaDistribution
                else:
                    fea_infos = performances[dataset_name][method_name[0]].FeaDistribution
                for snr in fea_infos:
                    fea_info = fea_infos[snr]
                    if isinstance(method_name, str):
                        save_path = os.path.join(save_dir, f'FeaDistribution_{dataset_name}_{snr:02d}_{method_name}.pdf')
                        self._draw_fea_distribution(fea_info['classes'], fea_info['feas'],
                                                    fea_info['gts'], fea_info['centers'], save_path, method_name)
                    else:
                        save_path = os.path.join(save_dir,
                                                 f'FeaDistribution_{dataset_name}_{snr:02d}_{method_name[0]}.pdf')
                        self._draw_fea_distribution(fea_info['classes'], fea_info['feas'],
                                                    fea_info['gts'], fea_info['centers'], save_path, method_name[1])

