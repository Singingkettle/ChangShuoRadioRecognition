import os

import matplotlib.pyplot as plt
import numpy as np

from .utils import load_json_log
from ..builder import LOSSACCURACIES
from ...common.utils import glob

plt.rcParams["font.family"] = "Times New Roman"


def get_new_fig(fn, fig_size=None):
    """ Init graphics """
    if fig_size is None:
        fig_size = [9, 9]
    fig = plt.figure(fn, fig_size)
    ax = fig.gca()  # Get Current Axis
    ax.cla()  # clear existing plot
    return fig, ax


def plot_train_curve(log_infos, legend, save_path, legend_config):
    fig, ax = get_new_fig('Training Curve', [8, 8])
    ax.set_xlim(1, 100)
    ax.set_ylim(0.8, 4.5)
    for i, log_info in enumerate(log_infos):
        log_dict = log_info['log_dict']
        method_name = log_info['name']
        train_metrics = log_info['train_metrics']
        legend_suffix = log_info['legend_suffix']
        epochs = list(log_dict.keys())
        for j, metric in enumerate(train_metrics):
            xs = []
            ys = []
            for epoch in epochs:
                iters = log_dict[epoch]['iter']
                if log_dict[epoch]['mode'][-1] == 'val':
                    iters = iters[:-1]
                # if epoch % 10 == 0:
                xs.append(epoch)
                ys.append(log_dict[epoch][metric][len(iters) - 1])
            xs = np.array(xs)
            ys = np.array(ys)
            if legend_suffix[j] is not None:
                legend_name = method_name + '-' + legend_suffix[j]
            else:
                legend_name = method_name

            ax.plot(
                xs, ys, label=legend_name, linewidth=0.6,
                color=legend_config[legend[legend_name]]['color'],
                # linestyle=legend_config[legend[legend_name]]['linestyle'],
                # marker=legend_config[legend[legend_name]]['marker'],
                # markersize=5,
            )

    leg = ax.legend(loc='upper right', prop={'size': 14, 'weight': 'bold'})
    leg.get_frame().set_edgecolor('black')
    ax.set_xlabel('Epochs', fontsize=18, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=18, fontweight='bold')
    ax.set_title('Training Curve', fontsize=18, fontweight='bold')

    # Don't allow the axis to be on top of your data
    ax.set_axisbelow(True)
    plt.tight_layout()  # set layout slim
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_test_curve(log_infos, legend, save_path, legend_config):
    fig, ax = get_new_fig('Test Curve', [8, 8])
    ax.set_xlim(10, 100)
    ax.set_ylim(0.2, 0.5)
    for i, log_info in enumerate(log_infos):
        log_dict = log_info['log_dict']
        method_name = log_info['name']
        test_metrics = log_info['test_metrics']
        legend_suffix = log_info['legend_suffix']
        epochs = list(log_dict.keys())
        for j, metric in enumerate(test_metrics):
            xs = []
            ys = []
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
                xs, ys, label=legend_name, linewidth=0.6,
                color=legend_config[legend[legend_name]]['color'],
                # linestyle=legend_config[legend[legend_name]]['linestyle'],
                # marker=legend_config[legend[legend_name]]['marker'],
                # markersize=5,
            )

    leg = ax.legend(loc='upper left', prop={'size': 14, 'weight': 'bold'})
    leg.get_frame().set_edgecolor('black')
    ax.set_xlabel('Epochs', fontsize=18, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=18, fontweight='bold')
    ax.set_title('Validation Curve', fontsize=18, fontweight='bold')

    # Don't allow the axis to be on top of your data
    ax.set_axisbelow(True)
    plt.tight_layout()  # set layout slim
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


@LOSSACCURACIES.register_module()
class LossAccuracyPlot(object):
    def __init__(self, name, method, log_dir, legend, legend_config=None):
        self.log_infos = []
        self.name = name
        self.method = method
        self.log_dir = log_dir
        self.legend = legend
        self.legend_config = legend_config

        if isinstance(self.method, list):
            for method_config in self.method:
                json_info = self.find_json_log(**method_config)
                self.log_infos.append(json_info)
        elif isinstance(self.method, dict):
            json_info = self.find_json_log(**self.method)
            self.log_infos.append(json_info)
        else:
            raise ValueError('The variable of method must be list!')

    def find_json_log(self, config, name, train_metrics=None, test_metrics=None, legend_suffix=None):
        if legend_suffix is None:
            legend_suffix = [None]
        if test_metrics is None:
            test_metrics = ['final']
        if train_metrics is None:
            train_metrics = ['loss']
        json_paths = glob(os.path.join(self.log_dir, config), '.json')
        # Assume that the last json file is right version
        json_paths = sorted(json_paths)
        log_dict = load_json_log(json_paths[-1])

        return dict(log_dict=log_dict, name=name, train_metrics=train_metrics, test_metrics=test_metrics,
                    legend_suffix=legend_suffix)

    def plot(self, save_dir):
        train_curve_save_path = os.path.join(save_dir, 'train_' + self.name)
        print('Save: ' + train_curve_save_path)
        plot_train_curve(self.log_infos, self.legend,
                         train_curve_save_path, self.legend_config)

        test_curve_save_path = os.path.join(save_dir, 'test_' + self.name)
        print('Save: ' + test_curve_save_path)
        plot_test_curve(self.log_infos, self.legend,
                        test_curve_save_path, self.legend_config)


if __name__ == '__main__':
    method_list = [
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
    from ..config.legend_config import LegendConfig

    legend_config_list = LegendConfig(18)
    legend_list = {
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
    confusion = LossAccuracyPlot('/home/citybuster/Data/SignalProcessing/Workdir',
                                 'cnn2_deepsig_iq_201610A.pdf', legend_list, method_list, legend_config_list)
    confusion.run(
        '/home/citybuster/Data/SignalProcessing/Workdir/cnn2_deepsig_iq_201610A/fig/')
