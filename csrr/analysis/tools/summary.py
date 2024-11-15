#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Project: ChangShuoRadioRecognition
File: summary.py
Author: Citybuster
Time: 2021/5/31 20:02
Email: chagshuo@bupt.edu.cn
"""

import os

from ..builder import TABLES


@TABLES.register_module()
class ModulationSummary:
    def __init__(self, dataset, legend_config=None, scatter_config=None):
        self.dataset = dataset
        self.legend_config = legend_config
        self.scatter_config = scatter_config

    def __call__(self, performances, save_dir):

        content = '# Summary of all Algorithms  \n'
        for dataset_name in self.dataset:
            content += f'## Experimental results of dataset {dataset_name}  \n'
            content += f'### SNR Accuracy Table  \n'
            row_sign = True
            for method_name in self.dataset[dataset_name]:
                if row_sign:
                    table_str1 = '| '
                    table_str2 = '|:---------------------------------------------:'
                    for snr in performances[dataset_name][method_name].snrs:
                        table_str1 = table_str1 + '| {:d}dB'.format(int(snr))
                        table_str2 = table_str2 + '|:---------------------------------------------:'
                    table_str1 = table_str1 + '| MAA |  \n'
                    table_str2 = table_str2 + '|:---------------------------------------------:|  \n'
                    content += table_str1 + table_str2
                    row_sign = False

                line_method = '|'
                line_method = line_method + method_name
                for acc in performances[dataset_name][method_name].ACC:
                    line_method = line_method + '| {:.3f}'.format(acc)
                line_method = line_method + '|  \n'
                content += line_method

            content += f'### Modulation F1 Score Table  \n'
            row_sign = True
            for method_name in self.dataset[dataset_name]:
                if row_sign:
                    table_str1 = '| '
                    table_str2 = '|:---------------------------------------------:'
                    for class_name in performances[dataset_name][method_name].classes:
                        table_str1 = table_str1 + '| {:d}dB'.format(int(class_name))
                        table_str2 = table_str2 + '|:---------------------------------------------:'
                    table_str1 = table_str1 + '| MAF |  \n'
                    table_str2 = table_str2 + '|:---------------------------------------------:|  \n'
                    content += table_str1 + table_str2
                    row_sign = False

                line_method = '|'
                line_method = line_method + method_name
                for f1 in performances[dataset_name][method_name].f1_score:
                    line_method = line_method + '| {:.3f}'.format(f1)
                line_method = line_method + '|  \n'
                content += line_method

        save_path = os.path.join(save_dir, 'summary.md')
        with open(save_path, 'w') as f:
            f.write(content)
