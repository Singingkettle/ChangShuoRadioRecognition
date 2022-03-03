import os
import os.path as osp
import pickle
import time
import zlib
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import zmq
from scipy.special import softmax
from torch.utils.data import Dataset
from tqdm import tqdm
from wtisp.common.fileio import dump as IODump
from wtisp.common.fileio import load as IOLoad
from wtisp.dataset.merge import get_merge_weight_by_search, get_merge_weight_by_optimization
from wtisp.dataset.amc_data import WTIMCDataset
from wtisp.apis import set_random_seed, train_task
from wtisp.common import get_root_logger, collect_env
from wtisp.common.fileio import del_files
from wtisp.common.utils import DictAction, Config, mkdir_or_exist
from wtisp.dataset import build_dataset
from wtisp.models import build_task
from wtisp.runner import init_dist, get_dist_info



def merge(results_dict, dataset, save_dir):
    pre_matrix = []
    for key_str in results_dict:
        pre_data = results_dict[key_str]
        pre_data = dataset._reshape_result(pre_data, len(dataset.CLASSES))
        pre_data = softmax(pre_data, axis=1)
        pre_data = pre_data[None, :, :]
        pre_matrix.append(pre_data)

    pre_matrix = np.concatenate(pre_matrix, axis=0)
    pre_max_index = np.argmax(pre_matrix, axis=2)
    pre_max_index = np.sum(pre_max_index, axis=0)
    gt_max_index = np.argmax(dataset.targets, axis=1) * len(pre_matrix)
    no_zero_index = np.nonzero((pre_max_index - gt_max_index))[0]

    bad_pre_matrix = pre_matrix[:, no_zero_index[:], :]
    targets = dataset.targets[no_zero_index[:], :]

    #
    optimization_start_time = time.time()
    w = get_merge_weight_by_optimization(bad_pre_matrix, targets)
    optimization_time = time.time() - optimization_start_time
    print(w)
    merge_matrix = np.dot(w.T, np.reshape(pre_matrix, (len(results_dict), -1)))
    merge_matrix = np.reshape(merge_matrix, (-1, len(dataset.CLASSES)))
    eval_results, eval_f1 = dataset._evaluate_mod(merge_matrix, prefix='final/', return_f1=True)

    ## This part of code is only for the ablation study about HCGDNN, which should be commented in the usual way.
    print('\n====================================================================================\n')
    print(
        'time:{:f}, accuracy:{:f}, f1:{:f}'.format(optimization_time, eval_results['final/snr_mean_all'], eval_f1))
    print('\n=====================================================================================\n')

    search_step_list = [0.3, 0.09, 0.027]
    for search_step in search_step_list:
        search_start_time = time.time()
        search_weight_list = get_merge_weight_by_search(len(results_dict), search_step)
        cur_max_accuracy = 0
        cur_f1 = 0
        cur_search_weight = None
        save_merge_matrix = None
        for search_weight in search_weight_list:
            search_weight = np.array(search_weight)
            search_weight = np.reshape(search_weight, (1, -1))
            tmp_merge_matrix = np.dot(search_weight, np.reshape(pre_matrix, (len(results_dict), -1)))
            tmp_merge_matrix = np.reshape(tmp_merge_matrix, (-1, len(dataset.CLASSES)))
            tmp_eval_results, tmp_f1 = dataset._evaluate_mod(tmp_merge_matrix, prefix='tmp/', return_f1=True)
            if cur_max_accuracy < tmp_eval_results['tmp/snr_mean_all']:
                cur_max_accuracy = tmp_eval_results['tmp/snr_mean_all']
                save_merge_matrix = tmp_merge_matrix
                cur_f1 = tmp_f1
                cur_search_weight = search_weight
        print(cur_search_weight)
        search_time = time.time() - search_start_time
        print('\n====================================================================================\n')
        print('time:{:f}, accuracy:{:f}, f1:{:f}'.format(search_time, cur_max_accuracy, cur_f1))
        print('\n====================================================================================\n')
        save_path = osp.join(save_dir, str(search_step) + '_pre.npy')
        np.save(save_path, save_merge_matrix)


configs = ['hcgdnn_abl_cg1_no_share_deepsig_iq_201610A',
           'hcgdnn_abl_cg2_no_share_deepsig_iq_201610A',
           'hcgdnn_abl_g1g2_no_share_deepsig_iq_201610A',
           'hcgdnn_abl_cg1g2_no_share_deepsig_iq_201610A']

work_dir = '/home/citybuster/Data/SignalProcessing/Workdir_Old'
for config in configs:
    print(config)
    cfg = Config.fromfile(config+'.py')
    cfg.data.test['use_cache'] = False
    dataset = build_dataset(cfg.data.test)
    # Get all results
    format_out_dir = os.path.join(work_dir, config, 'format_out')
    res_files = glob(os.path.join(format_out_dir, '*_pre.npy'))

    results_dict = dict()
    for res_file in res_files:
        if 'cnn' in res_file:
            results_dict['cnn'] = np.load(res_file)
        elif 'gru1' in res_file:
            results_dict['gru1'] = np.load(res_file)
        elif 'gru2' in res_file:
            results_dict['gru2'] = np.load(res_file)
        else:
            continue

    merge(results_dict, dataset, format_out_dir)


