import copy
import os

import matplotlib.pyplot as plt
import numpy as np

from .utils import (load_amc_evaluation_results, reorder_results,
                    radar_factory, get_classification_accuracy_and_f1)
from ..builder import ACCURACYF1S

plt.rcParams["font.family"] = "Times New Roman"


def get_new_fig(fn, fig_size=None):
    """ Init graphics """
    if fig_size is None:
        fig_size = [9, 9]
    fig = plt.figure(fn, fig_size)
    ax = fig.gca()  # Get Current Axis
    ax.cla()  # clear existing performance
    return fig, ax


def plot_curve_of_snr_score(snr_score, legend, save_path, legend_config, y_label, title):
    fig, ax = get_new_fig('Curve', [8, 10])
    SNRS = snr_score[0]['SNRS']
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
    snr_score = sorted(snr_score, key=lambda d: d['average'], reverse=True)
    for i, snr_accuracy in enumerate(snr_score):
        score = snr_accuracy['score']
        average_score = snr_accuracy['average']
        method_name = snr_accuracy['name']

        ys = np.array(score)
        max_list.append(score)
        legend_name = method_name + ' [{:.4f}]'.format(average_score)

        ax.plot(
            xs, ys, label=legend_name, linewidth=0.5,
            color=legend_config[legend[method_name]]['color'],
            linestyle=legend_config[legend[method_name]]['linestyle'],
            marker=legend_config[legend[method_name]]['marker'],
            markersize=3,
        )

    ## MLDNN
    # leg = ax.legend(loc='lower right', prop={'size': 10, 'weight': 'bold'})

    ## ZhangRuiYun
    # leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
    #                 prop={'size': 13, 'weight': 'bold'}, fancybox=True, ncol=3)

    ## HCGDNN
    leg = ax.legend(loc='lower right', prop={'size': 14, 'weight': 'bold'})

    leg.get_frame().set_edgecolor('black')
    ax.set_xlabel('SNRs', fontsize=18, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=18, fontweight='bold')
    ax.set_title(title, fontsize=18, fontweight='bold')

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
    plt.close(fig)


def plot_curve_of_class_score(class_scores, legend, save_path, legend_config, title, reorder=True, ):
    if reorder:
        class_scores = reorder_results(class_scores)

    fig, ax = get_new_fig('Curve', [8, 10])
    CLASSES = class_scores[0]['CLASSES']
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
    class_scores = sorted(class_scores, key=lambda d: d['average'], reverse=True)
    for i, class_score in enumerate(class_scores):
        score = class_score['score']
        average = class_score['average']
        method_name = class_score['name']

        ys = np.array(score)
        max_list.append(score)

        legend_name = method_name + ' [{:.4f}]'.format(average)

        ax.plot(
            xs, ys, label=legend_name, linewidth=0.5,
            color=legend_config[legend[method_name]]['color'],
            linestyle=legend_config[legend[method_name]]['linestyle'],
            marker=legend_config[legend[method_name]]['marker'],
            markersize=3,
        )

    ## MLDNN
    # leg = ax.legend(loc='lower left', prop={'size': 10, 'weight': 'bold'})

    ## ZhangRuiYun
    # leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
    #                 prop={'size': 13, 'weight': 'bold'}, fancybox=True, ncol=3)

    ## HCGDNN
    leg = ax.legend(loc='lower left', prop={'size': 14, 'weight': 'bold'})

    leg.get_frame().set_edgecolor('black')
    ax.set_xlabel('Modulations', fontsize=18, fontweight='bold')
    ax.set_ylabel('F1', fontsize=18, fontweight='bold')
    ax.set_title(title, fontsize=18, fontweight='bold')

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
    plt.close(fig)


def plot_radar_chart_of_class_score(class_scores, legend, save_path, legend_config, title, reorder=False):
    if reorder:
        class_scores = reorder_results(class_scores)

    CLASSES = class_scores[0]['CLASSES']
    print(CLASSES)
    theta = radar_factory(len(CLASSES), frame='polygon')
    fig, ax = plt.subplots(figsize=(8, 8), nrows=1, ncols=1, subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)

    f1s_list = []
    legend_list = []
    color_dict = dict(CNN='red', BiGRU1='green', BiGRU2='blue')
    for i, modulation_f1 in enumerate(class_scores):
        score = modulation_f1['score']
        average = modulation_f1['average']
        method_name = modulation_f1['name']
        legend_name = method_name + ' [{:.3f}]'.format(average)
        f1s_list.append(score)
        legend_list.append(legend_name)
        if method_name in color_dict:
            ax.plot(theta, score, color=color_dict[method_name])
            ax.fill(theta, score, facecolor=color_dict[method_name], alpha=0.25)
        else:
            ax.plot(theta, score, color=legend_config[legend[method_name]]['color'], linewidth=0.1)
            ax.fill(theta, score, facecolor=legend_config[legend[method_name]]['color'], alpha=0.25)
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], angle=-45, fontsize=18)
    ax.set_thetagrids(np.degrees(theta), CLASSES, fontsize=18)
    leg = ax.legend(legend_list, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                    prop={'size': 18, 'weight': 'bold'}, handletextpad=0.2,
                    markerscale=20, ncol=4, columnspacing=0.2)
    leg.get_frame().set_edgecolor('black')
    ax.set_title(title, fontsize=24, fontweight='bold')
    plt.tight_layout()  # set layout slim
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


@ACCURACYF1S.register_module()
class AccuracyF1Plot(object):
    def __init__(self, name, method, log_dir, legend, legend_config, extra_predictions=None):
        self.name = name
        self.method = method
        self.log_dir = log_dir
        self.legend = legend
        self.legend_config = legend_config
        self.extra_predictions = extra_predictions
        self.SNRS = None
        self.CLASSES = None
        self.snr_score = dict()
        self.class_score = dict()

        amc_results = load_amc_evaluation_results(self)
        for amc_result in amc_results:
            confusion_matrix = amc_result['cm']
            CLASSES = amc_results['cl']
            SNRS = amc_result['sn']
            method_name = amc_result['method_name']
            res = get_classification_accuracy_and_f1(method_name, confusion_matrix, SNRS, CLASSES)
            self.snr_score['all_class'].append(res[0])
            for class_key in res[1]:
                if class_key in self.snr_score:
                    self.snr_score[class_key].append(copy.copy(res[1][class_key]))
                else:
                    self.snr_score[class_key] = [copy.copy(res[1][class_key])]
            self.class_score['all_snr'].append(res[2])
            for snr_key in res:
                if snr_key in self.class_score['snr_' + snr_key]:
                    self.class_score['snr_' + snr_key].append(copy.copy(res[3][snr_key]))
                else:
                    self.class_score['snr_' + snr_key] = [copy.copy(res[3][snr_key])]

    def plot(self, save_dir):
        for key_str in self.snr_score:
            if 'all' in key_str:
                save_path = os.path.join(save_dir, 'snr_accuracy_curve_of_all_modulation-' + self.name)
                y_label = 'Accuracy'
                title = 'SNR VS Accuracy of ALL Modulation'
            else:
                save_path = os.path.join(save_dir, 'snr_f1_curve_of_' + key_str + '-' + self.name)
                y_label = 'F1'
                title = 'SNR VS F1 of Modulation ' + key_str
            print('Save: ' + save_path)
            plot_curve_of_snr_score(self.snr_score[key_str], self.legend, save_path, self.legend_config, y_label, title)

        for key_str in self.class_score:
            if 'all' in key_str:
                save_path = os.path.join(save_dir, 'modulation_f1_curve_all_snr-' + self.name)
                title = 'Modulation VS F1 of ALL SNR'.format(key_str)
            else:
                save_path = os.path.join(save_dir, 'modulation_f1_curve_of_' + key_str + '-' + self.name)
                title = 'Modulation VS F1 of SNR {}dB'.format(key_str)
            print('Save: ' + save_path)
            plot_curve_of_class_score(
                self.class_score[key_str], self.legend, save_path, self.legend_config, title)

            if 'all' in key_str:
                save_path = os.path.join(save_dir, 'modulation_f1_radar_chart_all_snr-' + self.name)
                title = 'Modulation VS F1 of ALL SNR'.format(key_str)
            else:
                save_path = os.path.join(save_dir, 'modulation_f1_radar_chart_of_' + key_str + '-' + self.name)
                title = 'Modulation VS F1 of SNR {}dB'.format(key_str)
            print('Save: ' + save_path)
            plot_radar_chart_of_class_score(
                self.class_score[key_str], self.legend, save_path, self.legend_config, title)


