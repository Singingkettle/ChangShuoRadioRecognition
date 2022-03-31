#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Project: wtisignalprocessing
File: summary.py
Author: Citybuster
Time: 2021/5/31 20:02
Email: chagshuo@bupt.edu.cn
"""

import os

from .utils import load_amc_evaluation_results, get_classification_accuracy_and_f1_for_plot
from ..builder import SUMMARIES


@SUMMARIES.register_module()
class ModulationSummary(object):
    def __init__(self, log_dir, name, dataset=None, config_legend_map=None, config_method_map=None):
        self.log_dir = log_dir
        self.name = name
        self.config_legend_map = config_legend_map
        self.config_method_map = config_method_map
        self.SNRS = None
        self.CLASSES = None
        self.methods_dict = dict()
        for config in config_method_map.keys():
            dataset_name = '201610A'
            for name in dataset:
                if name in config:
                    dataset_name = name
                    break
            if dataset_name in self.methods_dict:
                self.methods_dict[dataset_name].append(self.config_method_map[config])
            else:
                self.methods_dict[dataset_name] = [self.config_method_map[config]]

        self.summary_info = dict()
        for dataset_name in self.methods_dict.keys():
            self.SNRS = None
            self.CLASSES = None
            self.method = self.methods_dict[dataset_name]
            snr_accuracies, modulation_f1s = self.get_accuracy_f1()
            self.summary_info[dataset_name] = dict(acc=snr_accuracies, f1=modulation_f1s, snrs=self.SNRS,
                                                   classes=self.CLASSES)

    def get_accuracy_f1(self):
        snr_accuracies = list()
        modulation_f1s = list()
        amc_results = load_amc_evaluation_results(self)
        for amc_result in amc_results:
            confusion_matrix = amc_result['cm']
            CLASSES = amc_results['cl']
            SNRS = amc_result['sn']
            method_name = amc_result['method_name']
            snr_accuracy, modulation_f1 = get_classification_accuracy_and_f1_for_plot(method_name,
                                                                                      confusion_matrix, SNRS,
                                                                                      CLASSES)
            snr_accuracies.append(snr_accuracy)
            modulation_f1s.append(modulation_f1)

        return snr_accuracies, modulation_f1s

    def plot_summary(self, save_path):
        with open(save_path, 'w') as f:
            f.write('# Summary of all Algorithms  \n')
            for dataset_name in self.summary_info.keys():
                f.write('## Experimental results of dataset {}  \n'.format(dataset_name))

                # SNR Accuracy Table
                f.write('### SNR Accuracy Table of dataset {}  \n'.format(dataset_name))
                table_str1 = '| '
                table_str2 = '|:---------------------------------------------:'
                for snr in self.summary_info[dataset_name]['snrs']:
                    table_str1 = table_str1 + '| {:d}dB'.format(int(snr))
                    table_str2 = table_str2 + '|:---------------------------------------------:'
                table_str1 = table_str1 + '| MAA |  \n'
                table_str2 = table_str2 + '|:---------------------------------------------:|  \n'
                f.write(table_str1)
                f.write(table_str2)

                for method_index, method in enumerate(self.methods_dict[dataset_name]):
                    line_method = '|'
                    line_method = line_method + method['name']
                    for acc in self.summary_info[dataset_name]['acc'][method_index]['accs']:
                        line_method = line_method + '| {:.3f}'.format(acc)
                    line_method = line_method + '| {:.3f}'.format(
                        self.summary_info[dataset_name]['acc'][method_index]['average_accuracy'])
                    line_method = line_method + '|  \n'
                    f.write(line_method)

                # Modulation F1 Score Table
                f.write('### Modulation F1 Score Table of dataset {}  \n'.format(dataset_name))
                table_str1 = '| '
                table_str2 = '|:---------------------------------------------:'
                for modulation in self.summary_info[dataset_name]['classes']:
                    table_str1 = table_str1 + '| {}'.format(modulation)
                    table_str2 = table_str2 + '|:---------------------------------------------:'
                table_str1 = table_str1 + '| MAF |  \n'
                table_str2 = table_str2 + '|:---------------------------------------------:|  \n'
                f.write(table_str1)
                f.write(table_str2)

                for method_index, method in enumerate(self.methods_dict[dataset_name]):
                    line_method = '|'
                    line_method = line_method + method['name']
                    for f1 in self.summary_info[dataset_name]['f1'][method_index]['f1s']:
                        line_method = line_method + '| {:.3f}'.format(f1)
                    line_method = line_method + '| {:.3f}'.format(
                        self.summary_info[dataset_name]['f1'][method_index]['average_f1'])
                    line_method = line_method + '|  \n'
                    f.write(line_method)

    def plot(self, save_dir):
        save_path = os.path.join(save_dir, 'summary' + self.name)
        print('Save: ' + save_path)
        self.plot_summary(save_path=save_path)
