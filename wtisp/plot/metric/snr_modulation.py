import copy
import os

import matplotlib.pyplot as plt
import numpy as np

from wtisp.common.fileio import load as IOLoad
from ..builder import SNRMODULATIONS

plt.rcParams["font.family"] = "Times New Roman"


def get_new_fig(fn, figsize=[9, 9]):
    """ Init graphics """
    fig = plt.figure(fn, figsize)
    ax = fig.gca()  # Get Current Axis
    ax.cla()  # clear existing plot
    return fig, ax


def plot_snr_accuracy_curve(snr_accuracys, legends, save_path, legend_configs):
    fig, ax = get_new_fig('Curve', [8, 8])
    SNRS = snr_accuracys[0]['SNRS']
    xs = np.array([i for i in range(len(SNRS))]) / (len(SNRS) - 1)
    ax.set_xticks(xs)  # values
    xs_str = ['%9d' % i for i in SNRS]
    ax.set_xticklabels(xs_str)  # labels
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    max_list = []
    for i, snr_accuracy in enumerate(snr_accuracys):
        accs = snr_accuracy['accs']
        average_accuracy = snr_accuracy['average_accuracy']
        method_name = snr_accuracy['name']

        ys = np.array(accs)
        max_list.append(accs)
        legend_name = method_name + ' [{:.3f}]'.format(average_accuracy)

        ax.plot(
            xs, ys, label=legend_name, linewidth=1,
            color=legend_configs[legends[method_name]]['color'],
            linestyle=legend_configs[legends[method_name]]['linestyle'],
            marker=legend_configs[legends[method_name]]['marker'],
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
    # Customize the minor grid
    ax.grid(b=True, which='minor', linestyle=':',
            linewidth='0.5', color='black', alpha=0.5)

    plt.tick_params(which='minor', bottom=False,
                    top=False, left=False, right=False)

    ax.tick_params(which='minor', bottom=False,
                   top=False, left=False, right=False)
    plt.setp(ax.get_xticklabels(), rotation=50,
             horizontalalignment='right')
    plt.tight_layout()  # set layout slim
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_modulation_f1_curve(modulation_f1s, legends, save_path, legend_configs, reorder=True):
    def reorder_results(modulation_f1s):
        if len(modulation_f1s) == 1:
            min_modulation_f1s = copy.deepcopy(modulation_f1s[0]['f2s'])

        else:
            num_modulations = len(modulation_f1s[0]['f2s'])
            num_methods = len(modulation_f1s)
            min_modulation_f1s = copy.deepcopy(modulation_f1s[0]['f2s'])
            for modulation_index in range(num_modulations):
                for method_index in range(1, num_methods):
                    if min_modulation_f1s[modulation_index] > modulation_f1s[method_index]['f2s'][modulation_index]:
                        min_modulation_f1s[modulation_index] = copy.copy(
                            modulation_f1s[method_index]['f2s'][modulation_index])
        sort_indices = np.argsort(np.array(min_modulation_f1s) * -1)

        new_modulation_f1s = []
        num_methods = len(modulation_f1s)
        for method_index in range(num_methods):
            new_f2s = list()
            new_CLASSES = list()
            new_modulation_f1 = dict()
            for modulation_index in sort_indices:
                new_f2s.append(copy.copy(modulation_f1s[method_index]['f2s'][modulation_index]))
                new_CLASSES.append(copy.copy(modulation_f1s[method_index]['CLASSES'][modulation_index]))
            new_modulation_f1['f2s'] = copy.deepcopy(new_f2s)
            new_modulation_f1['CLASSES'] = copy.deepcopy(new_CLASSES)
            new_modulation_f1['average_f2'] = copy.deepcopy(modulation_f1s[method_index]['average_f2'])
            new_modulation_f1['name'] = copy.deepcopy(modulation_f1s[method_index]['name'])
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
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    max_list = []
    for i, modulation_f1 in enumerate(modulation_f1s):
        f2s = modulation_f1['f2s']
        average_f2 = modulation_f1['average_f2']
        method_name = modulation_f1['name']

        ys = np.array(f2s)
        max_list.append(f2s)

        legend_name = method_name + ' [{:.3f}]'.format(average_f2)

        ax.plot(
            xs, ys, label=legend_name, linewidth=1,
            color=legend_configs[legends[method_name]]['color'],
            linestyle=legend_configs[legends[method_name]]['linestyle'],
            marker=legend_configs[legends[method_name]]['marker'],
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
    # Customize the minor grid
    ax.grid(b=True, which='minor', linestyle=':',
            linewidth='0.5', color='black', alpha=0.5)

    plt.tick_params(which='minor', bottom=False,
                    top=False, left=False, right=False)

    ax.tick_params(which='minor', bottom=False,
                   top=False, left=False, right=False)
    plt.setp(ax.get_xticklabels(), rotation=25,
             horizontalalignment='right')
    plt.tight_layout()  # set layout slim
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


@SNRMODULATIONS.register_module()
class SNRModulationCurve(object):
    def __init__(self, log_dir, name, legends, methods, legend_configs=None):
        self.log_dir = log_dir
        self.name = name
        self.legends = legends
        self.methods = methods
        self.legend_configs = legend_configs
        self.SNRS = None
        self.CLASSES = None
        self.snr_accuracys, self.modulation_f1s = self.load_methods()

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        annos = IOLoad(ann_file)
        SNRS = annos['SNRS']
        CLASSES = annos['CLASSES']
        ann_info = annos['ANN']
        mods_dict = annos['mods_dict']
        snrs_dict = annos['snrs_dict']
        replace_dict = {'PAM4': '4PAM', 'QAM16': '16QAM', 'QAM64': '64QAM'}
        for index, item in enumerate(CLASSES):
            if item in replace_dict:
                CLASSES[index] = replace_dict[item]
        return SNRS, CLASSES, mods_dict, snrs_dict, ann_info

    def load_methods(self, ):
        snr_accuracys = list()
        modulation_f1s = list()
        for method in self.methods:
            config = method['config']
            name = method['name']
            if 'has_snr_classifier' in method:
                has_snr_classifier = method['has_snr_classifier']
            else:
                has_snr_classifier = False

            format_out_dir = os.path.join(self.log_dir, config, 'format_out')
            if has_snr_classifier:
                results = np.load(os.path.join(
                    format_out_dir, 'merge_pre.npy'))
            else:
                results = np.load(os.path.join(
                    format_out_dir, 'pre.npy'))

            SNRS, CLASSES, mods_dict, snrs_dict, ann_info = self.load_annotations(
                os.path.join(format_out_dir, 'ann.json'))
            if self.SNRS is None or self.CLASSES is None:
                self.SNRS = SNRS
                self.CLASSES = CLASSES

            if (self.SNRS == SNRS) and (self.CLASSES == CLASSES):
                confusion_matrix = np.zeros(
                    (len(SNRS), len(CLASSES), len(CLASSES)), dtype=np.float64)

                for idx in range(len(ann_info)):
                    ann = ann_info[idx]
                    snrs = ann['snrs']
                    labels = ann['mod_labels']
                    if len(snrs) == 1 and len(labels) == 1:
                        predict_class_index = int(
                            np.argmax(results[idx, :]))
                        confusion_matrix[snrs_dict['{:.3f}'.format(
                            snrs[0])], labels[0], predict_class_index] += 1
                    else:
                        raise ValueError(
                            'Please check your dataset, the size of snrs and labels are both 1 for any item. '
                            'However, the current item with the idx {:d} has the snrs size {:d} and the '
                            'labels size {:d}'.format(idx, snrs.size, labels.size))

                accs = list()
                for snr_index, snr in enumerate(SNRS):
                    conf = confusion_matrix[snr_index, :, :]
                    cor = np.sum(np.diag(conf))
                    ncor = np.sum(conf) - cor
                    accs.append(1.0 * cor / (cor + ncor))

                conf = np.sum(confusion_matrix, axis=0)
                cor = np.sum(np.diag(conf))
                ncor = np.sum(conf) - cor
                average_accuracy = 1.0 * cor / (cor + ncor)

                snr_accuracy = dict(
                    accs=accs, average_accuracy=average_accuracy, name=name, SNRS=SNRS)
                snr_accuracys.append(snr_accuracy)

                f2s = list()
                for i in range(len(CLASSES)):
                    f2 = 2.0 * conf[i, i] / \
                         (np.sum(conf[i, :]) + np.sum(conf[:, i]))
                    f2s.append(f2)
                average_f2 = sum(f2s) / float(len(CLASSES))
                modulation_f1 = dict(
                    f2s=f2s, average_f2=average_f2, name=name, CLASSES=CLASSES)
                modulation_f1s.append(modulation_f1)
            else:
                raise ValueError(
                    'Please check your input methods. They should be evaluated in the same dataset with the same configuration.')

        return snr_accuracys, modulation_f1s

    def plot(self, save_dir):
        save_path = os.path.join(save_dir, 'snr_accuracy_' + self.name)
        print('Save: ' + save_path)
        plot_snr_accuracy_curve(
            self.snr_accuracys, self.legends, save_path, self.legend_configs)
        save_path = os.path.join(save_dir, 'modulation_f1_' + self.name)
        print('Save: ' + save_path)
        plot_modulation_f1_curve(
            self.modulation_f1s, self.legends, save_path, self.legend_configs)


if __name__ == '__main__':
    methods = [
        dict(
            config='cnn2_deepsig_iq_201610A',
            name='CNN2-IQ',
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
    confusion = SNRModulationCurve('/home/citybuster/Data/SignalProcessing/Workdir',
                                   'cnn2_deepsig_iq_201610A.pdf', legends, methods, legend_configs)
    confusion.plot('./')
