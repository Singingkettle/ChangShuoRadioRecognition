import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from ..builder import TRAINTESTCURVES

plt.rcParams["font.family"] = "Times New Roman"


def get_new_fig(fn, figsize=[9, 9]):
    """ Init graphics """
    fig = plt.figure(fn, figsize)
    ax = fig.gca()  # Get Current Axis
    ax.cla()  # clear existing plot
    return fig, ax


def plot_train_curve(log_infos, legend, save_path, legend_configs):
    fig, ax = get_new_fig('Training Curve', [8, 8])
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 3)
    for i, log_info in enumerate(log_infos):
        log_dict = log_info['log_dict']
        method_name = log_info['name']
        train_metrics = log_info['train_metrics']
        legend_suffix = log_info['legend_suffix']
        epochs = list(log_dict.keys())
        for j, metric in enumerate(train_metrics):
            xs = []
            ys = []
            num_iters_per_epoch = log_dict[epochs[0]]['iter'][-1]
            for epoch in epochs:
                iters = log_dict[epoch]['iter']
                if log_dict[epoch]['mode'][-1] == 'val':
                    iters = iters[:-1]
                if epoch % 5 == 0:
                    xs.append(epoch)
                    ys.append(log_dict[epoch][metric][len(iters) - 1])
            xs = np.array(xs)
            ys = np.array(ys)
            if legend_suffix[j] is not None:
                legend_name = method_name + '-' + legend_suffix[j]
            else:
                legend_name = method_name

            ax.plot(
                xs, ys, label=legend_name, linewidth=1,
                color=legend_configs[legend[legend_name]]['color'],
                linestyle=legend_configs[legend[legend_name]]['linestyle'],
                marker=legend_configs[legend[legend_name]]['marker'],
                markersize=5,
            )

    leg = ax.legend(loc='upper right', prop={'size': 15, 'weight': 'bold'})
    leg.get_frame().set_edgecolor('black')
    ax.set_xlabel('Epochs', fontsize=18, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=18, fontweight='bold')
    ax.set_title('Training Curve', fontsize=18, fontweight='bold')

    # Don't allow the axis to be on top of your data
    ax.set_axisbelow(True)

    # Turn on the minor TICKS, which are required for the minor GRID
    ax.minorticks_on()

    # Customize the major grid
    ax.grid(b=True, which='major', linestyle='-',
            linewidth='0.5', color='black', alpha=0.2)
    # # Customize the minor grid
    # ax.grid(b=True, which='minor', linestyle=':',
    #         linewidth='0.5', color='black', alpha=0.5)

    plt.tick_params(which='minor', bottom=False,
                    top=False, left=False, right=False)
    plt.tick_params(which='major', bottom=True,
                    top=False, left=True, right=False)

    ax.tick_params(which='minor', bottom=False,
                   top=False, left=False, right=False)
    ax.tick_params(which='major', bottom=True,
                   top=False, left=True, right=False)

    plt.tight_layout()  # set layout slim
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_test_curve(log_infos, legend, save_path, legend_configs):
    fig, ax = get_new_fig('Test Curve', [8, 8])
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 1)
    for i, log_info in enumerate(log_infos):
        log_dict = log_info['log_dict']
        method_name = log_info['name']
        test_metrics = log_info['test_metrics']
        legend_suffix = log_info['legend_suffix']
        epochs = list(log_dict.keys())
        for j, metric in enumerate(test_metrics):
            xs = []
            ys = []
            num_iters_per_epoch = log_dict[epochs[0]]['iter'][-1]
            for epoch in epochs:
                if log_dict[epoch]['mode'][-1] == 'val':
                    xs.append(epoch)
                    ys.append(log_dict[epoch][metric])
            xs = np.array(xs)
            ys = np.array(ys)
            if legend_suffix[j] is not None:
                legend_name = method_name + '-' + legend_suffix[j]
            else:
                legend_name = method_name

            ax.plot(
                xs, ys, label=legend_name, linewidth=1,
                color=legend_configs[legend[legend_name]]['color'],
                linestyle=legend_configs[legend[legend_name]]['linestyle'],
                marker=legend_configs[legend[legend_name]]['marker'],
                markersize=5,
            )

    leg = ax.legend(loc='lower right', prop={'size': 15, 'weight': 'bold'})
    leg.get_frame().set_edgecolor('black')
    ax.set_xlabel('Epochs', fontsize=18, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=18, fontweight='bold')
    ax.set_title('Test Curve', fontsize=18, fontweight='bold')

    # Don't allow the axis to be on top of your data
    ax.set_axisbelow(True)

    # Turn on the minor TICKS, which are required for the minor GRID
    ax.minorticks_on()

    # Customize the major grid
    ax.grid(b=True, which='major', linestyle='-',
            linewidth='0.5', color='black', alpha=0.2)
    # # Customize the minor grid
    # ax.grid(b=True, which='minor', linestyle=':',
    #         linewidth='0.5', color='black', alpha=0.5)

    plt.tick_params(which='minor', bottom=False,
                    top=False, left=False, right=False)
    plt.tick_params(which='major', bottom=True,
                    top=False, left=True, right=False)

    ax.tick_params(which='minor', bottom=False,
                   top=False, left=False, right=False)
    ax.tick_params(which='major', bottom=True,
                   top=False, left=True, right=False)

    plt.tight_layout()  # set layout slim
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


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


@TRAINTESTCURVES.register_module()
class LossAccuracyCurve(object):
    def __init__(self, log_dir, name, legends, methods, legend_configs=None):
        self.log_dir = log_dir
        self.log_infos = []
        self.name = name
        self.legends = legends
        self.legend_configs = legend_configs

        if isinstance(methods, list):
            for method_config in methods:
                json_info = self.find_json_log(**method_config)
                self.log_infos.append(json_info)
        else:
            raise ValueError('The variable of methods must be list!')

    def find_json_log(self, config, name, has_snr_classifier=False):
        json_paths = glob.glob(os.path.join(
            self.log_dir, config, '*.json'))
        # assume that the last json file is right version
        json_paths = sorted(json_paths)
        log_dict = load_json_log(json_paths[-1])
        if has_snr_classifier:
            train_metrics = ['loss_high', 'loss_low', 'loss_merge', 'loss_snr']
            test_metrics = ['merge/snr_mean_all',
                            'high/snr_mean_all', 'low/snr_mean_all', 'snr/snr_acc']
            legend_suffix = [None, 'High', 'Low', 'SNR']
        else:
            train_metrics = ['loss_cls']
            test_metrics = ['common/snr_mean_all']
            legend_suffix = [None]

        return dict(log_dict=log_dict, name=name, train_metrics=train_metrics, test_metrics=test_metrics,
                    legend_suffix=legend_suffix)

    def plot(self, save_dir):
        train_curve_save_path = os.path.join(save_dir, 'train_' + self.name)
        print('Save: ' + train_curve_save_path)
        plot_train_curve(self.log_infos, self.legends,
                         train_curve_save_path, self.legend_configs)

        test_curve_save_path = os.path.join(save_dir, 'test_' + self.name)
        print('Save: ' + test_curve_save_path)
        plot_test_curve(self.log_infos, self.legends,
                        test_curve_save_path, self.legend_configs)


if __name__ == '__main__':
    methods = [
        dict(
            config='cnn2_deepsig_iq_201610A',
            name='CNN2-IQ',
            has_snr_classifier=False,
        ),
        dict(
            config='cnn3_deepsig_iq_201610A',
            name='CNN3-IQ',
            has_snr_classifier=False,
        ),
    ]
    from .legend_config import LegendConfig

    legend_configs = LegendConfig(18)
    legends = {
        'MLDNN': 0,
        'CLDNN-IQ': 1,
        'CLDNN-AP': 2,
        'CNN2-IQ': 3,
        'CNN2-AP': 4,
        'CNN3-IQ': 5,
        'CNN3-AP': 6,
        'CNN4-IQ': 7,
        'CNN4-AP': 8,
        'DensCNN': 9,
        'ResCNN': 10,
        'MLDNN-IQ': 11,
        'MLDNN-AP': 12,
        'MLDNN-GRU': 13,
        'MLDNN-Last': 14,
        'MLDNN-Add': 15,
        'MLDNN-Att': 16,
        'MLDNN-Grade': 17,
    }
    confusion = LossAccuracyCurve('/home/citybuster/Data/SignalProcessing/Workdir',
                                  'cnn2_deepsig_iq_201610A.pdf', legends, methods, legend_configs)
    confusion.run(
        '/home/citybuster/Data/SignalProcessing/Workdir/cnn2_deepsig_iq_201610A/fig/')
