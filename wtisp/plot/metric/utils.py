#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Project: wtisignalprocessing
File: utils.py
Author: Citybuster
Time: 2021/5/31 21:45
Email: chagshuo@bupt.edu.cn
"""
import os

import numpy as np

from wtisp.common.fileio import load as IOLoad


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
    return SNRS, CLASSES, mods_dict, snrs_dict, ann_info


def load_method(obj_pointer):
    snr_accuracies = list()
    modulation_f1s = list()
    for method in obj_pointer.method:
        config = method['config']
        name = method['name']
        if 'has_snr_classifier' in method:
            has_snr_classifier = method['has_snr_classifier']
        else:
            has_snr_classifier = False

        format_out_dir = os.path.join(obj_pointer.log_dir, config, 'format_out')
        if has_snr_classifier:
            results = np.load(os.path.join(
                format_out_dir, 'merge_pre.npy'))
        else:
            results = np.load(os.path.join(
                format_out_dir, 'pre.npy'))

        SNRS, CLASSES, mods_dict, snrs_dict, ann_info = load_annotation(os.path.join(format_out_dir, 'ann.json'))
        if obj_pointer.SNRS is None or obj_pointer.CLASSES is None:
            obj_pointer.SNRS = SNRS
            obj_pointer.CLASSES = CLASSES

        if (obj_pointer.SNRS == SNRS) and (obj_pointer.CLASSES == CLASSES):
            confusion_matrix = np.zeros(
                (len(SNRS), len(CLASSES), len(CLASSES)), dtype=np.float64)

            for idx in range(len(ann_info)):
                ann = ann_info[idx]
                snrs = ann['snrs']
                labels = ann['mod_labels']
                if len(snrs) == 1 and len(labels) == 1:
                    predict_class_index = int(
                        np.argmax(results[idx, :]))
                    confusion_matrix[snrs_dict['{:.3f}'.format(
                        snrs[0])], labels[0], predict_class_index] += 1
                else:
                    raise ValueError(
                        'Please check your dataset, the size of snrs and labels are both 1 for any item. '
                        'However, the current item with the idx {:d} has the snrs size {:d} and the '
                        'labels size {:d}'.format(idx, snrs.size, labels.size))

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
                accs=accs, average_accuracy=average_accuracy, name=name, SNRS=SNRS)
            snr_accuracies.append(snr_accuracy)

            f1s = list()
            for i in range(len(CLASSES)):
                f1 = 2.0 * conf[i, i] / \
                     (np.sum(conf[i, :]) + np.sum(conf[:, i]))
                f1s.append(f1)
            average_f1 = sum(f1s) / float(len(CLASSES))
            modulation_f1 = dict(
                f1s=f1s, average_f1=average_f1, name=name, CLASSES=CLASSES)
            modulation_f1s.append(modulation_f1)
        else:
            raise ValueError(
                'Please check your input methods. They should be evaluated '
                'in the same dataset with the same configuration.')

    return snr_accuracies, modulation_f1s
