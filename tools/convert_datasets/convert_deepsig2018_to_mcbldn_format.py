#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Project: wtisignalprocessing
File: convert_deepsig2018_to_mcbldn_format.py
Author: Citybuster
Time: 2021/12/13 13:05
Email: chagshuo@bupt.edu.cn
"""

import os.path as osp
import sys
from concurrent import futures
from glob import glob

import numpy as np


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar

    Args:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """

    format_str = "{0:." + str(decimals) + "f}"
    percents = format_str.format(100 * (iteration / float(total)))
    filledLength = int(round(bar_length * iteration / float(total)))
    bar = '' * filledLength + '-' * (bar_length - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' %
                     (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()


def generate_by_mcbldn(data):
    file_name = osp.basename(data)
    data = np.load(data)
    constellation = np.zeros((8, 128, 128), dtype=np.float32)

    x_max = np.max(np.abs(data[0, :])) * 1.01
    y_max = np.max(np.abs(data[1, :])) * 1.01
    x_indexes = np.floor(data[0, :] / x_max * 64) + 64
    y_indexes = np.floor(data[1, :] / y_max * 64) + 64
    x_indexes = x_indexes.astype(np.int)
    y_indexes = y_indexes.astype(np.int)

    for point_index in range(x_indexes.shape[0]):
        slot_index = point_index // 128
        constellation[slot_index, x_indexes[point_index], y_indexes[point_index]] += 1

    np.save(osp.join('/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201801A/mcbldn',
                     file_name), constellation)


if __name__ == '__main__':
    datas = glob(
        osp.join('/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201801A/sequence_data/iq/',
                 '*npy'))

    num_items = len(datas)
    with futures.ProcessPoolExecutor(max_workers=24) as executor:
        fs = [executor.submit(generate_by_mcbldn, item) for item in datas]
        for i, f in enumerate(futures.as_completed(fs)):
            # Write progress to error so that it can be seen
            print_progress(i, num_items, prefix='Convert DeepSig2018 To MCBLDN Format', suffix='Done ', bar_length=40)
    # for item in datas:
    #     generate_by_mcbldn(item)
