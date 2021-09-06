#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Project: wtisignalprocessing
File: filter_config.py
Author: Citybuster
Time: 2021/5/13 14:24
Email: chagshuo@bupt.edu.cn
"""
import glob
import json
import os
import os.path as osp
from collections import defaultdict


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


def get_the_best_checkpoint(log_dir, config, has_snr_classifier):
    json_paths = glob.glob(os.path.join(log_dir, config, '*.json'))
    # assume that the last json file is right version
    json_paths = sorted(json_paths)
    log_dict = load_json_log(json_paths[-1])
    if has_snr_classifier:
        metric = 'merge/snr_mean_all'
    else:
        metric = 'common/snr_mean_all'

    epochs = list(log_dict.keys())
    accuracy = 0
    best_epoch = 0
    for epoch in epochs:
        if log_dict[epoch]['mode'][-1] == 'val':
            if log_dict[epoch][metric][0] > accuracy:
                accuracy = log_dict[epoch][metric][0]
                best_epoch = epoch

    return best_epoch


def filter_config_by_rule(cfg, is_test=True, is_regeneration=False):
    configs = dict()
    # Add config from confusion_maps
    if 'confusion_maps' in cfg.plot:
        if isinstance(cfg.plot['confusion_maps'], dict):
            config = cfg.plot['confusion_maps']['config']
            if 'has_snr_classifier' in cfg.plot['confusion_maps']:
                has_snr_classifier = cfg.plot['confusion_maps']['has_snr_classifier']
            else:
                has_snr_classifier = False
            if config in configs:
                configs[config] = (configs[config] or has_snr_classifier)
            else:
                configs[config] = has_snr_classifier

        elif isinstance(cfg.plot['confusion_maps'], list):
            for confusion_map in cfg.plot['confusion_maps']:
                config = confusion_map['config']
                if 'has_snr_classifier' in confusion_map:
                    has_snr_classifier = confusion_map['has_snr_classifier']
                else:
                    has_snr_classifier = False
                if config in configs:
                    configs[config] = (configs[config] or has_snr_classifier)
                else:
                    configs[config] = has_snr_classifier
        else:
            raise ValueError('The confusion maps must be list or dict!')

    # Add config from train_test_curves
    if 'train_test_curves' in cfg.plot:
        train_test_curves = cfg.plot['train_test_curves']
        if isinstance(train_test_curves, dict):
            for method in train_test_curves['methods']:
                if 'has_snr_classifier' in method:
                    has_snr_classifier = method['has_snr_classifier']
                else:
                    has_snr_classifier = False

                if method['config'] in configs:
                    configs[method['config']] = (
                            configs[method['config']] or has_snr_classifier)
                else:
                    configs[method['config']] = has_snr_classifier
        elif isinstance(train_test_curves, list):
            for train_test_curve in train_test_curves:
                for method in train_test_curve['methods']:
                    if 'has_snr_classifier' in method:
                        has_snr_classifier = method['has_snr_classifier']
                    else:
                        has_snr_classifier = False

                    if method['config'] in configs:
                        configs[method['config']] = (
                                configs[method['config']] or has_snr_classifier)
                    else:
                        configs[method['config']] = has_snr_classifier
        else:
            raise ValueError('The train test curves must be list or dict!')

    # Add config from snr_modulation
    if 'snr_modulation' in cfg.plot:
        snr_modulation = cfg.plot['snr_modulation']
        if isinstance(snr_modulation, dict):
            for method in snr_modulation['methods']:
                if 'has_snr_classifier' in method:
                    has_snr_classifier = method['has_snr_classifier']
                else:
                    has_snr_classifier = False

                if method['config'] in configs:
                    configs[method['config']] = (
                            configs[method['config']] or has_snr_classifier)
                else:
                    configs[method['config']] = has_snr_classifier
        elif isinstance(snr_modulation, list):
            for snr_accuracy in snr_modulation:
                for method in snr_accuracy['methods']:
                    if 'has_snr_classifier' in method:
                        has_snr_classifier = method['has_snr_classifier']
                    else:
                        has_snr_classifier = False

                    if method['config'] in configs:
                        configs[method['config']] = (
                                configs[method['config']] or has_snr_classifier)
                    else:
                        configs[method['config']] = has_snr_classifier
        else:
            raise ValueError('The snr_modulation must be list or dict!')

    if is_test:
        save_configs = dict()
        for config, has_snr_classifier in configs.items():
            if osp.isdir(osp.join(cfg.log_dir, config)):
                format_out_dir = osp.join(cfg.log_dir, config, 'format_out')
                json_paths = glob.glob(osp.join(format_out_dir, '*.json'))
                npy_paths = glob.glob(osp.join(format_out_dir, '*.npy'))
                if not is_regeneration:
                    if (len(json_paths) == 1) and ((len(npy_paths) == 1) or (
                            len(npy_paths) == 4)) and (not is_regeneration):
                        continue
            if 'feature_based' in config:
                save_configs[config] = -1
            else:
                best_epoch = get_the_best_checkpoint(
                    cfg.log_dir, config, has_snr_classifier)
                save_configs[config] = best_epoch

        return save_configs
    else:
        return configs
