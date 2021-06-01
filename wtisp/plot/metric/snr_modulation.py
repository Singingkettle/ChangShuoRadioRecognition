import copy
import os

import matplotlib.pyplot as plt
import numpy as np

from .utils import load_method
from ..builder import SNRMODULATIONS

plt.rcParams["font.family"] = "Times New Roman"


def get_new_fig(fn, fig_size=None):
    """ Init graphics """
    if fig_size is None:
        fig_size = [9, 9]
    fig = plt.figure(fn, fig_size)
    ax = fig.gca()  # Get Current Axis
    ax.cla()  # clear existing plot
    return fig, ax


def plot_snr_accuracy_curve(snr_accuracies, legend, save_path, legend_config):
    fig, ax = get_new_fig('Curve', [8, 8])
    SNRS = snr_accuracies[0]['SNRS']
    xs = np.array([i for i in range(len(SNRS))]) / (len(SNRS) - 1)
    xs_str = ['%9d' % i for i in SNRS]
    ax.set_xticks(xs)  # values
    ax.set_xticklabels(xs_str)  # labels

    ys = np.array([i for i in range(11)]) / 10
    ys_str = ['%.1f' % i for i in ys]
    ax.set_yticks(ys)  # values
    ax.set_yticklabels(ys_str)  # labels
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    max_list = []
    for i, snr_accuracy in enumerate(snr_accuracies):
        accs = snr_accuracy['accs']
        average_accuracy = snr_accuracy['average_accuracy']
        method_name = snr_accuracy['name']

        ys = np.array(accs)
        max_list.append(accs)
        legend_name = method_name + ' [{:.3f}]'.format(average_accuracy)

        ax.plot(
            xs, ys, label=legend_name, linewidth=1,
            color=legend_config[legend[method_name]]['color'],
            linestyle=legend_config[legend[method_name]]['linestyle'],
            marker=legend_config[legend[method_name]]['marker'],
            markersize=5,
        )

    # big_indexes = []
    # for i in range(len(max_list[0])):
    #     big_index = 0
    #     big_val = max_list[0][i]
    #     for j in range(len(max_list)):
    #         if big_val < max_list[j][i]:
    #             big_val = max_list[j][i]
    #             big_index = j
    #     big_indexes.append(big_index)

    # # print('\\hline')
    # for j, accs in enumerate(max_list):
    #     p_str = snr_accuracys[j]['name']
    #     for i, p in enumerate(accs):
    #         if big_indexes[i] == j:
    #             p_str = p_str + '&\\textbf{%.3f}' % p
    #         else:
    #             p_str = p_str + '&%.3f' % p
    #     # print(p_str + '\\\\')

    leg = ax.legend(loc='lower right', prop={'size': 15, 'weight': 'bold'})
    leg.get_frame().set_edgecolor('black')
    ax.set_xlabel('SNRs', fontsize=18, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=18, fontweight='bold')
    ax.set_title('SNR-Accuracy Curve', fontsize=18, fontweight='bold')

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
    plt.setp(ax.get_xticklabels(), rotation=50,
             horizontalalignment='right')
    plt.tight_layout()  # set layout slim
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_modulation_f1_curve(modulation_f1s, legend, save_path, legend_config, reorder=True):
    def reorder_results(f1s):
        if len(f1s) == 1:
            min_modulation_f1s = copy.deepcopy(f1s[0]['f1s'])

        else:
            num_modulations = len(f1s[0]['f1s'])
            num_method = len(f1s)
            min_modulation_f1s = copy.deepcopy(f1s[0]['f1s'])
            for modulation_index in range(num_modulations):
                for method_index in range(1, num_method):
                    if min_modulation_f1s[modulation_index] > f1s[method_index]['f1s'][modulation_index]:
                        min_modulation_f1s[modulation_index] = copy.copy(
                            f1s[method_index]['f1s'][modulation_index])
        sort_indices = np.argsort(np.array(min_modulation_f1s) * -1)

        new_modulation_f1s = []
        num_method = len(f1s)
        for method_index in range(num_method):
            new_f1s = list()
            new_CLASSES = list()
            new_modulation_f1 = dict()
            for modulation_index in sort_indices:
                new_f1s.append(copy.copy(f1s[method_index]['f1s'][modulation_index]))
                new_CLASSES.append(copy.copy(f1s[method_index]['CLASSES'][modulation_index]))
            new_modulation_f1['f1s'] = copy.deepcopy(new_f1s)
            new_modulation_f1['CLASSES'] = copy.deepcopy(new_CLASSES)
            new_modulation_f1['average_f1'] = copy.deepcopy(f1s[method_index]['average_f1'])
            new_modulation_f1['name'] = copy.deepcopy(f1s[method_index]['name'])
            new_modulation_f1s.append(new_modulation_f1)
        return new_modulation_f1s

    if reorder:
        modulation_f1s = reorder_results(modulation_f1s)

    fig, ax = get_new_fig('Curve', [8, 8])
    CLASSES = modulation_f1s[0]['CLASSES']
    xs = np.array([i for i in range(len(CLASSES))]) / (len(CLASSES) - 1)
    xs_str = ['%9s' % i for i in CLASSES]
    ax.set_xticks(xs)  # values
    ax.set_xticklabels(xs_str)  # labels

    ys = np.array([i for i in range(11)]) / 10
    ys_str = ['%.1f' % i for i in ys]
    ax.set_yticks(ys)  # values
    ax.set_yticklabels(ys_str)  # labels
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    max_list = []
    for i, modulation_f1 in enumerate(modulation_f1s):
        f1s = modulation_f1['f1s']
        average_f1 = modulation_f1['average_f1']
        method_name = modulation_f1['name']

        ys = np.array(f1s)
        max_list.append(f1s)

        legend_name = method_name + ' [{:.3f}]'.format(average_f1)

        ax.plot(
            xs, ys, label=legend_name, linewidth=1,
            color=legend_config[legend[method_name]]['color'],
            linestyle=legend_config[legend[method_name]]['linestyle'],
            marker=legend_config[legend[method_name]]['marker'],
            markersize=5,
        )

    # big_indexes = []
    # for i in range(len(max_list[0])):
    #     big_index = 0
    #     big_val = max_list[0][i]
    #     for j in range(len(max_list)):
    #         if big_val < max_list[j][i]:
    #             big_val = max_list[j][i]
    #             big_index = j
    #     big_indexes.append(big_index)

    # print('\\hline')
    # for j, accs in enumerate(max_list):
    #     p_str = modulation_f1s[j]['name']
    #     for i, p in enumerate(accs):
    #         if big_indexes[i] == j:
    #             p_str = p_str + '&\\textbf{%.3f}' % p
    #         else:
    #             p_str = p_str + '&%.3f' % p
    #     print(p_str + '\\\\')

    leg = ax.legend(loc='lower left', prop={'size': 15, 'weight': 'bold'})
    leg.get_frame().set_edgecolor('black')
    ax.set_xlabel('Modulations', fontsize=18, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=18, fontweight='bold')
    ax.set_title('Modulation-F1 Score Curve', fontsize=18, fontweight='bold')

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
    plt.setp(ax.get_xticklabels(), rotation=25,
             horizontalalignment='right')
    plt.tight_layout()  # set layout slim
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


@SNRMODULATIONS.register_module()
class SNRModulationCurve(object):
    def __init__(self, log_dir, name, legend, method, legend_config=None):
        self.log_dir = log_dir
        self.name = name
        self.legend = legend
        self.method = method
        self.legend_config = legend_config
        self.SNRS = None
        self.CLASSES = None
        self.snr_accuracies, self.modulation_f1s = load_method(self)

    def plot(self, save_dir):
        save_path = os.path.join(save_dir, 'snr_accuracy_' + self.name)
        print('Save: ' + save_path)
        plot_snr_accuracy_curve(
            self.snr_accuracies, self.legend, save_path, self.legend_config)
        save_path = os.path.join(save_dir, 'modulation_f1_' + self.name)
        print('Save: ' + save_path)
        plot_modulation_f1_curve(
            self.modulation_f1s, self.legend, save_path, self.legend_config)


if __name__ == '__main__':
    method_list = [
        dict(
            config='cnn2_deepsig_iq_201610A',
            name='CNN2-IQ',
            has_snr_classifier=False,
        ),
    ]
    from .legend_config import LegendConfig

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
    confusion = SNRModulationCurve('/home/citybuster/Data/SignalProcessing/Workdir',
                                   'cnn2_deepsig_iq_201610A.pdf', legend_list, method_list, legend_config_list)
    confusion.plot('./')
