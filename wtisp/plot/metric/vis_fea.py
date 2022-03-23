#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Project: wtisignalprocessing
File: vis_fea.py
Author: Citybuster
Time: 2021/8/22 16:34
Email: chagshuo@bupt.edu.cn
"""

import copy
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from tsnecuda import TSNE

from .utils import load_annotation
from ..builder import VISFEATURES

plt.rcParams["font.family"] = "Times New Roman"


def get_new_fig(fn, fig_size=None):
    """ Init graphics """
    if fig_size is None:
        fig_size = [9, 9]
    fig1 = plt.figure(fn, fig_size)
    ax1 = fig1.gca()  # Get Current Axis
    ax1.cla()  # clear existing plot
    return fig1, ax1


def plot_fea_distribution(fea, label_names, scatter_config, save_path, fig_size=None):
    if fig_size is None:
        fig_size = [10, 11]
    print('Save: ' + save_path)

    fea_dict = dict()
    for idx, label in enumerate(label_names):
        if label in fea_dict:
            fea_dict[label].append(fea[idx, :][None, :])
        else:
            fea_dict[label] = [fea[idx, :][None, :]]

    fig, ax = get_new_fig('Features Visualization Results', fig_size)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-12, 10)
    for label in fea_dict:
        data = np.concatenate(fea_dict[label], axis=0)
        ax.scatter(data[:, 0], data[:, 1], s=0.06, c=scatter_config[label]['color'],
                   marker=scatter_config[label]['marker'], label=label, cmap='tab10')

    leg = ax.legend(loc='lower center', prop={'size': 15, 'weight': 'bold'},
                    handletextpad=0.2, markerscale=20, ncol=6, columnspacing=0.2)
    leg.get_frame().set_edgecolor('black')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


@VISFEATURES.register_module()
class VisFea(object):
    def __init__(self, log_dir, name, method, scatter_config=None, reduction='tsne'):
        self.log_dir = log_dir
        self.name = name
        self.config = method['config']
        self.reduction = reduction
        self.format_out_dir = os.path.join(self.log_dir, self.config, 'format_out')
        self.label_names = self._get_label_names()
        self.fea_dict = self._get_fea_from_file()
        self.scatter_config = scatter_config

    def _get_label_names(self):
        _, _, mods_dict, _, ann_info = load_annotation(os.path.join(self.format_out_dir, 'ann.json'))
        index_class_dict = {index: mod for mod, index in mods_dict.items()}
        label_names = []
        for ann in ann_info:
            ## TODO: Support multi-label for single data
            item = index_class_dict[ann['mod_labels'][0]]
            label_names.append(copy.copy(item))

        return label_names

    def _get_fea_from_file(self):
        fea_files = glob.glob(self.format_out_dir + '/*fea.npy')
        fea_dict = dict()
        if len(fea_files) > 0:
            for fea_file in fea_files:
                fea = np.load(fea_file)
                fea = getattr(self, self.reduction)(fea)
                fea = fea / (np.max(np.abs(fea)) + np.finfo(np.float64).eps) * 9.9
                fea_name = os.path.basename(fea_file)
                fea_name = fea_name.split('.')[0]
                fea_dict[fea_name] = fea

        return fea_dict

    def pca(self, x):
        try:
            pca = PCA(n_components=2)
            x_2d = pca.fit_transform(x)
            return x_2d
        except Exception as e:
            print(e)
            raise ValueError('Something wrong with PCA! Please check your input fea!')

    def tsne(self, x):
        try:
            tsne = TSNE(n_components=2, random_seed=0)
            x_2d = tsne.fit_transform(x)
            return x_2d
        except Exception as e:
            print(e)
            raise ValueError('Something wrong with TSNE! Please check your input fea!')

    def origin(self, x):
        if x.shape[1] == 2:
            return x
        else:
            raise ValueError('The origin fea must be a N*2 matrix!')

    def plot(self, save_dir):
        for fea in self.fea_dict:
            save_path = os.path.join(save_dir, fea + '_' + self.name)
            plot_fea_distribution(self.fea_dict[fea], self.label_names, self.scatter_config, save_path=save_path)


if __name__ == '__main__':
    pass
