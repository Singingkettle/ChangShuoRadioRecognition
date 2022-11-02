#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Project: ChangShuoRadioRecognition
File: utils.py
Author: Citybuster
Time: 2021/5/31 21:45
Email: chagshuo@bupt.edu.cn
"""
import copy
import json
import os
from collections import defaultdict

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


def get_classification_accuracy_and_f1(prediction_name, confusion_matrix, SNRS, CLASSES):
    all_class_snr_accuracy = list()
    single_class_snr_f1 = {class_name: [] for class_name in CLASSES}
    single_snr_class_f1 = {snr: [] for snr in SNRS}
    for snr_index, snr in enumerate(SNRS):
        conf = confusion_matrix[snr_index, :, :]
        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor
        all_class_snr_accuracy.append(1.0 * cor / (cor + ncor))
        for class_index in range(len(CLASSES)):
            num_tp_fn = confusion_matrix[snr_index, class_index, :]
            num_tp_fp = confusion_matrix[snr_index, :, class_index]
            num_tp_tn = confusion_matrix[snr_index, class_index, class_index]
            class_snr_f1 = 2 * num_tp_tn / (num_tp_fn + num_tp_fp)
            single_class_snr_f1[CLASSES[class_index]].append(class_snr_f1)
            single_snr_class_f1[SNRS[snr_index]].append(class_snr_f1)

    average_accuracy = sum(all_class_snr_accuracy) / float(len(SNRS))
    all_class_snr_accuracy_info = dict(
        score=all_class_snr_accuracy, average=average_accuracy, name=prediction_name, SNRS=SNRS)

    single_class_snr_f1_info = dict()
    for class_name in single_class_snr_f1:
        average_f1 = sum(single_class_snr_f1[class_name]) / float(len(SNRS))
        info = dict(score=single_class_snr_f1[class_name], average=average_f1, name=prediction_name, SNRS=SNRS)
        single_class_snr_f1_info[class_name] = info

    all_snr_class_f1 = list()
    conf = np.sum(confusion_matrix, axis=0)
    for class_index in range(len(CLASSES)):
        f1 = 2.0 * conf[class_index, class_index] / (np.sum(conf[class_index, :]) + np.sum(conf[:, class_index]))
        all_snr_class_f1.append(f1)
    average_f1 = sum(all_snr_class_f1) / float(len(CLASSES))
    all_snr_class_f1_info = dict(score=all_snr_class_f1, average=average_f1, name=prediction_name, CLASSES=CLASSES)

    single_snr_class_f1_info = dict()
    for snr in single_snr_class_f1:
        average_f1 = sum(single_snr_class_f1[snr]) / float(len(CLASSES))
        info = dict(score=single_snr_class_f1[snr], average=average_f1, name=prediction_name, CLASSES=CLASSES)
        single_snr_class_f1_info[snr] = info

    return all_class_snr_accuracy_info, single_class_snr_f1_info, all_snr_class_f1_info, single_snr_class_f1_info


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


def reorder_results(class_scores):
    if len(class_scores) == 1:
        min_clss_scores = copy.deepcopy(class_scores[0]['score'])

    else:
        num_classes = len(class_scores[0]['score'])
        num_method = len(class_scores)
        min_clss_scores = copy.deepcopy(class_scores[0]['score'])
        for class_index in range(num_classes):
            for method_index in range(1, num_method):
                if min_clss_scores[class_index] > class_scores[method_index]['score'][class_index]:
                    min_clss_scores[class_index] = copy.copy(
                        class_scores[method_index]['score'][class_index])
    sort_indices = np.argsort(np.array(min_clss_scores) * -1)

    new_class_scores = []
    num_method = len(class_scores)
    for method_index in range(num_method):
        new_scores = list()
        new_classes = list()
        new_class_score = dict()
        for class_index in sort_indices:
            new_scores.append(copy.copy(class_scores[method_index]['score'][class_index]))
            new_classes.append(copy.copy(class_scores[method_index]['CLASSES'][class_index]))
        new_class_score['score'] = copy.deepcopy(new_scores)
        new_class_score['CLASSES'] = copy.deepcopy(new_classes)
        new_class_score['average'] = copy.deepcopy(class_scores[method_index]['average'])
        new_class_score['name'] = copy.deepcopy(class_scores[method_index]['name'])
        new_class_scores.append(new_class_score)
    return new_class_scores


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


def load_json_log(json_log):
    log_dict = dict()
    with open(json_log, 'r') as log_file:
        for line in log_file:
            log = json.loads(line.strip())
            # skip lines without `epoch` field
            if 'epoch' not in log:
                continue
            epoch = log.pop('epoch')
            if epoch not in log_dict:
                log_dict[epoch] = defaultdict(list)
            for k, v in log.items():
                log_dict[epoch][k].append(v)
    return log_dict
