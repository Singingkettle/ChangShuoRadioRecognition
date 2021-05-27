# Copyright (c) Shuo Chang. All Rights Reserved.

import argparse
import glob
import json
import os
import random

_DATA_VERSION = ['Chuan2021.04.22']


def parse_args():
    parser = argparse.ArgumentParser(description='Convert ChuanSEI Dataset')
    parser.add_argument('--data_root', type=str,
                        help='data root to deepsig data')
    args = parser.parse_args()

    return args


def generate_ann_info_json(data, mods, devs, data_num, save_path) -> object:
    ann_dict = dict(data=data, mods=mods, devs=devs, data_num=data_num)
    print('Save annotation json at {}'.format(save_path))
    json.dump(ann_dict, open(save_path, 'w'), indent=4, sort_keys=True)


def generate(root_dir, data_ratios):
    train_ann_info = list()
    val_ann_info = list()
    test_ann_info = list()
    modulation_dict = dict()
    device_dict = dict()

    train_num = 0
    val_num = 0
    test_num = 0
    modulation_num = 0
    device_num = 0
    for modulation in os.listdir(root_dir):
        modulation_dir = os.path.join(root_dir, modulation)
        if os.path.isdir(modulation_dir):
            if modulation not in modulation_dict:
                modulation_dict[modulation] = modulation_num
                modulation_num += 1
            for device in os.listdir(modulation_dir):
                device_dir = os.path.join(modulation_dir, device)
                if os.path.isdir(device_dir):
                    if device not in device_dict:
                        device_dict[device] = device_num
                        device_num += 1
                    raw_data_list = glob.glob(device_dir + '/' + ('[0-9]' * 5) + '.npy')
                    raw_data_list = [os.path.basename(file_path) for file_path in raw_data_list]
                    random.shuffle(raw_data_list)
                    tmp_train_num = int(len(raw_data_list) * data_ratios[0])
                    tmp_val_num = int(len(raw_data_list) * data_ratios[1])
                    for index, file_name in enumerate(raw_data_list):
                        ann = dict(mod=modulation_dict[modulation], dev=device_dict[device])
                        item = dict(ann=ann, filename=file_name)
                        if index < tmp_train_num:
                            train_ann_info.append(item)
                            train_num += 1
                        elif index < (tmp_train_num + tmp_val_num):
                            val_ann_info.append(item)
                            val_num += 1
                        else:
                            test_ann_info.append(item)
                            test_num += 1

    # Generate train json
    generate_ann_info_json(train_ann_info, modulation_dict, device_dict, train_num,
                           os.path.join(root_dir, 'train.json'))
    # Generate test json
    generate_ann_info_json(test_ann_info, modulation_dict, device_dict, test_num,
                           os.path.join(root_dir, 'test.json'))
    # Generate val json
    generate_ann_info_json(val_ann_info, modulation_dict, device_dict, val_num,
                           os.path.join(root_dir, 'val.json'))
    # Generate train+val json
    generate_ann_info_json(train_ann_info + val_ann_info, modulation_dict, device_dict, train_num + val_num,
                           os.path.join(root_dir, 'train_and_val.json'))


def main():
    args = parse_args()
    data_ratios = [0.6, 0.2, 0.2]  # train, val test
    for version in _DATA_VERSION:
        generate(os.path.join(args.data_root, version), data_ratios)


if __name__ == '__main__':
    main()
