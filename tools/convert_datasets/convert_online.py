#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Project: wtisignalprocessing
File: convert_online.py
Author: Citybuster
Time: 2021/11/4 14:39
Email: chagshuo@bupt.edu.cn
"""
import argparse
# Copyright (c) Shuo Chang. All Rights Reserved.
import json
import os
import os.path as osp

import numpy as np
import zmq
import copy


def parse_args():
    parser = argparse.ArgumentParser(
        description='Collect data online')
    parser.add_argument('--num', type=int, default=5000, help='the number of items for a specific modulation')
    parser.add_argument('--data_root', type=str,
                        help='data root to deepsig data')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    print('Command Line Args:', args)

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    tcp_address = 'tcp://10.112.54.27:28570'
    socket.connect(tcp_address)
    socket.setsockopt_string(zmq.SUBSCRIBE, '')

    if not osp.isdir(osp.join(args.data_root, 'Online', 'iq')):
        os.makedirs(osp.join(args.data_root, 'Online', 'iq'))

    num_data = 0
    mods_dict = {'bpsk': 0, 'qpsk': 1, '16qam': 2, '32qam': 3, '64qam': 4, '8psk': 5}

    modulation_count = {'bpsk': 0, 'qpsk': 0, '16qam': 0, '32qam': 0, '64qam': 0, '8psk': 0}
    modulation_annotations = {'bpsk': None, 'qpsk': None, '16qam': None, '32qam': None, '64qam': None, '8psk': None}
    while True:
        print(modulation_count)
        is_break = True
        for modulation in modulation_count:
            if modulation_count[modulation] < args.num:
                is_break = False
                break
        if is_break:
            break
        message = socket.recv()
        message_list = message.split(b'-')
        frame_id = message_list[0].decode('utf-8')
        true_modulation = message_list[1].decode('utf-8')
        print('{}-{}'.format(frame_id, true_modulation))
        if modulation_count[true_modulation] == args.num:
            print('Please start the next modulation collection!!!')
            continue

        start_point = len(message_list[0]) + len(message_list[1]) + 2
        data = message[start_point:]
        data = np.frombuffer(data, dtype=np.float32).reshape(2, -1)
        ann_info = {'labels': true_modulation}

        split_index = 0
        while split_index * 128 < data.shape[1]:
            num_data += 1
            file_name = '{:0>12d}.npy'.format(num_data)
            iq_path = osp.join(args.data_root, 'Online', 'iq', file_name)
            split_data = data[:, split_index*128:(split_index+1)*128]
            np.save(iq_path, split_data)

            modulation_count[true_modulation] += 1
            annotation_item = {'filename': file_name, 'ann': ann_info}
            if modulation_annotations[true_modulation] is None:
                modulation_annotations[true_modulation] = [annotation_item]
            else:
                modulation_annotations[true_modulation].append(annotation_item)
            split_index += 2

    train_annotations = []
    val_annotations = []
    train_num = args.num * 0.8
    for modulation in modulation_annotations:
        annotations = copy.deepcopy(modulation_annotations[modulation])
        train_annotations.extend(annotations[:train_num])
        val_annotations.extend(annotations[train_num:])

    train_data = {'data': train_annotations, 'mods': mods_dict}
    json.dump(train_data, open(osp.join(args.data_root, 'Online') + '/{}.json'.format('train'), 'w'),
              indent=4, sort_keys=True)

    val_data = {'data': val_annotations, 'mods': mods_dict}
    json.dump(val_data, open(osp.join(args.data_root, 'Online') + '/{}.json'.format('val'), 'w'),
              indent=4, sort_keys=True)


if __name__ == '__main__':
    main()
