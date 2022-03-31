#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Project: wtisignalprocessing
File: utils.py
Author: Citybuster
Time: 2021/5/31 21:45
Email: chagshuo@bupt.edu.cn
"""
import copy
import os

import numpy as np
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

from ...common.fileio import load as IOLoad


def load_annotation(ann_file):
    """Load annotation from annotation file."""
    annos = IOLoad(ann_file)
    SNRS = annos['SNRS']
    CLASSES = annos['CLASSES']
    ann_info = annos['ANN']
    mods_dict = annos['mods_dict']
    snrs_dict = annos['snrs_dict']
    replace_dict = {'PAM4': '4PAM', 'QAM16': '16QAM', 'QAM64': '64QAM'}
    for index, item in enumerate(CLASSES):
        if item in replace_dict:
            CLASSES[index] = replace_dict[item]
            mods_dict[replace_dict[item]] = mods_dict[item]
            mods_dict.pop(item)
    return SNRS, CLASSES, mods_dict, snrs_dict, ann_info


def get_classification_accuracy_and_f1_for_plot(prediction_name, confusion_matrix, SNRS, CLASSES):
    accs = list()
    for snr_index, snr in enumerate(SNRS):
        conf = confusion_matrix[snr_index, :, :]
        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor
        accs.append(1.0 * cor / (cor + ncor))

    conf = np.sum(confusion_matrix, axis=0)
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    average_accuracy = 1.0 * cor / (cor + ncor)

    snr_accuracy = dict(
        accs=accs, average_accuracy=average_accuracy, name=prediction_name, SNRS=SNRS)

    f1s = list()
    for i in range(len(CLASSES)):
        f1 = 2.0 * conf[i, i] / (np.sum(conf[i, :]) + np.sum(conf[:, i]))
        f1s.append(f1)
    average_f1 = sum(f1s) / float(len(CLASSES))
    modulation_f1 = dict(
        f1s=f1s, average_f1=average_f1, name=prediction_name, CLASSES=CLASSES)

    return snr_accuracy, modulation_f1


def load_amc_evaluation_results(obj_pointer):
    amc_results = list()

    if isinstance(obj_pointer.method, dict):
        obj_pointer.method = [obj_pointer.method]

    for method in obj_pointer.method:
        config = method['config']
        name = method['name']

        format_out_dir = os.path.join(obj_pointer.log_dir, config, 'format_out')
        pre_files = [os.path.join(format_out_dir, 'Final.pkl')]
        if 'extra_predictions' in method:
            for extra_prediction in method['extra_predictions']:
                pre_files.append(os.path.join(format_out_dir, extra_prediction + '.pkl'))
        for pre_file in pre_files:
            if os.path.isfile(pre_file):
                save_res = IOLoad(pre_file)
                prediction_name = os.path.basename(pre_file)
                if prediction_name is 'Final':
                    prediction_name = name
                save_res['method_name'] = prediction_name
                amc_results.append(save_res)

    return amc_results


def reorder_results(f1s):
    if len(f1s) == 1:
        min_modulation_f1s = copy.deepcopy(f1s[0]['f1s'])

    else:
        num_modulations = len(f1s[0]['f1s'])
        num_method = len(f1s)
        min_modulation_f1s = copy.deepcopy(f1s[0]['f1s'])
        for modulation_index in range(num_modulations):
            for method_index in range(1, num_method):
                if min_modulation_f1s[modulation_index] > f1s[method_index]['f1s'][modulation_index]:
                    min_modulation_f1s[modulation_index] = copy.copy(
                        f1s[method_index]['f1s'][modulation_index])
    sort_indices = np.argsort(np.array(min_modulation_f1s) * -1)

    new_modulation_f1s = []
    num_method = len(f1s)
    for method_index in range(num_method):
        new_f1s = list()
        new_CLASSES = list()
        new_modulation_f1 = dict()
        for modulation_index in sort_indices:
            new_f1s.append(copy.copy(f1s[method_index]['f1s'][modulation_index]))
            new_CLASSES.append(copy.copy(f1s[method_index]['CLASSES'][modulation_index]))
        new_modulation_f1['f1s'] = copy.deepcopy(new_f1s)
        new_modulation_f1['CLASSES'] = copy.deepcopy(new_CLASSES)
        new_modulation_f1['average_f1'] = copy.deepcopy(f1s[method_index]['average_f1'])
        new_modulation_f1['name'] = copy.deepcopy(f1s[method_index]['name'])
        new_modulation_f1s.append(new_modulation_f1)
    return new_modulation_f1s


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta
