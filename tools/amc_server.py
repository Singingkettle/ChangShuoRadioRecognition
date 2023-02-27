#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Project: ChangShuoRadioRecognition
File: amc_server.py
Author: Citybuster
Time: 2021/10/27 16:59
Email: chagshuo@bupt.edu.cn
"""

import argparse
import os.path as osp
import signal
import struct
import sys

import numpy as np
import torch
import zmq
from torch.utils.tensorboard import SummaryWriter

from csrr.common.utils import Config, fuse_conv_bn, mkdir_or_exist
from csrr.datasets import build_dataset
from csrr.models import build_method
from csrr.runner import load_checkpoint

_IS_SIG_UP = False


def parse_args():
    parser = argparse.ArgumentParser(
        description='ChangShuoRadioRecognition test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--log_dir', help='dir to format the log file for online figures results')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
             'the inference speed')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    args = parser.parse_args()

    return args


def data_normalization(a):
    a = (a - np.expand_dims(np.min(a, axis=1), axis=1)) / np.expand_dims((np.max(a, axis=1) - np.min(a, axis=1)),
                                                                         axis=1)
    return a


def sigint_handler(signum, frame):
    _IS_SIG_UP = True
    print('Receive an interrupt signal!')


def main():
    args = parse_args()
    print('Command Line Args:', args)

    cfg = Config.fromfile(args.config)
    print(cfg.pretty_text)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    # build the model and load checkpoint
    model = build_method(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    dataset = build_dataset(cfg.data.test)
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # log
    if args.log_dir is not None:
        # update figure_configs according to CLI args if args.work_dir is not None
        log_dir = osp.join(args.log_dir, osp.splitext(osp.basename(args.config))[0])
    else:
        # use figure_configs file_name as default work_dir if cfg.work_dir is None
        log_dir = osp.join('./online_performance_dirs', osp.splitext(osp.basename(args.config))[0])
    log_dir = osp.join(log_dir, 'tf_logs')
    mkdir_or_exist(log_dir)

    # safely exit
    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGHUP, sigint_handler)
    signal.signal(signal.SIGTERM, sigint_handler)

    log_writer = SummaryWriter(log_dir)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    tcp_address = 'tcp://*:5555'
    socket.bind(tcp_address)

    true_modulation_info = dict()
    pred_modulation_info = dict()

    frame_id = 1
    while True:
        message = socket.recv()
        data = struct.unpack('<2048d', message)
        data = [data[::2], data[1::2]]
        data = np.concatenate(data, axis=0)
        data = data_normalization(data)
        data = np.expand_dims(data, axis=0)
        data = np.reshape(data, (2, 1, -1))
        data = torch.from_numpy(data)
        data = data.to("cuda")
        with torch.no_grad():
            res = model.simple_test(**data)[0]
        index = np.argmax(res)
        socket.send_string('{:d}'.format(index))
        predict_modulation = model.CLASSES[index]
        print('frame id: {}, predict modulation: {}'.format(frame_id, predict_modulation))
        # if true_modulation in true_modulation_info:
        #     true_modulation_info[true_modulation] += 1
        # else:
        #     true_modulation_info[true_modulation] = 1
        #     pred_modulation_info[true_modulation] = 0
        #
        # if predict_modulation is true_modulation:
        #     pred_modulation_info[true_modulation] += 1
        #
        # dynamic_accuracy = pred_modulation_info[true_modulation] / true_modulation_info[true_modulation]
        # log_writer.add_scalar(true_modulation, dynamic_accuracy, true_modulation_info[true_modulation])

        if _IS_SIG_UP:
            print('Exit online evaluation!!!')
            log_writer.close()
            break

    sys.exit()


if __name__ == '__main__':
    main()
