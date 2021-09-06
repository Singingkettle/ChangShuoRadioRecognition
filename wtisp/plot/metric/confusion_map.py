# from ..builder import CONFUSIONS
import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from matplotlib.collections import QuadMesh
from pandas import DataFrame

from .utils import load_annotation
from ..builder import CONFUSIONS

plt.rcParams["font.family"] = "Times New Roman"


def get_new_fig(fn, fig_size=None):
    """ Init graphics """
    if fig_size is None:
        fig_size = [9, 9]
    fig1 = plt.figure(fn, fig_size)
    ax1 = fig1.gca()  # Get Current Axis
    ax1.cla()  # clear existing plot
    return fig1, ax1


def config_cell_text_and_colors(array_df, lin, col, o_text, face_colors, position, fz, show_null_values=0):
    """
      config cell text and colors
      and return text elements to add and to dell
    """
    text_add = []
    text_del = []
    cell_val = array_df[lin][col]
    tot_all = array_df[-1][-1]
    per = (float(cell_val) / tot_all) * 100
    curr_column = array_df[:, col]
    ccl = len(curr_column)

    # last line  and/or last column
    if (col == (ccl - 1)) or (lin == (ccl - 1)):
        # tots and percents
        per_ok = 0
        if cell_val != 0:
            if (col == ccl - 1) and (lin == ccl - 1):
                tot_rig = 0
                for i in range(array_df.shape[0] - 1):
                    tot_rig += array_df[i][i]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif col == ccl - 1:
                tot_rig = array_df[lin][lin]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif lin == ccl - 1:
                tot_rig = array_df[col][col]
                per_ok = (float(tot_rig) / cell_val) * 100
            per_err = 100 - per_ok
        else:
            per_ok = per_err = 0

        if per_ok == 100:
            per_ok_s = '100%'
        else:
            per_ok_s = '%.2f%%' % per_ok

        # text to DEL
        text_del.append(o_text)

        # text to ADD
        font_prop = fm.FontProperties(weight='bold', size=fz)
        text_kwargs = dict(color='black', ha="center", va="center",
                           gid='sum', fontproperties=font_prop)
        lis_txt = ['%d' % cell_val, per_ok_s, '%.2f%%' % per_err]
        lis_kwa = [text_kwargs]
        dic = text_kwargs.copy()
        dic['color'] = 'g'
        lis_kwa.append(dic)
        dic = text_kwargs.copy()
        dic['color'] = 'r'
        lis_kwa.append(dic)
        lis_pos = [(o_text._x, o_text._y - 0.3), (o_text._x, o_text._y), (o_text._x, o_text._y + 0.3)]
        for i in range(len(lis_txt)):
            new_text = dict(x=lis_pos[i][0], y=lis_pos[i]
            [1], text=lis_txt[i], kw=lis_kwa[i])
            # print 'lin: %s, col: %s, newText: %s' %(lin, col, newText)
            text_add.append(new_text)
        # print '\n'

        # set background color for sum cells (last line and last column)
        carr = [1, 1, 1, 1.0]
        if (col == ccl - 1) and (lin == ccl - 1):
            carr = [0.9, 0.9, 0.9, 1.0]
        face_colors[position] = carr

    else:
        if per > 0:
            txt = '%s\n%.2f%%' % (cell_val, per)
        else:
            if show_null_values == 0:
                txt = ''
            elif show_null_values == 1:
                txt = '0'
            else:
                txt = '0\n0.0%'
        o_text.set_text(txt)

        # main diagonal
        if col == lin:
            # set color of the text in the diagonal to white
            o_text.set_color('black')
            # set background color in the diagonal to blue
            face_colors[position] = [0.35, 0.8, 0.55, 1.0]
        else:
            o_text.set_color('black')

    return text_add, text_del


def insert_totals(df_cm):
    """ insert total column and line (the last ones) """
    sum_col = []
    for c in df_cm.columns:
        sum_col.append(df_cm[c].sum())
    sum_lin = []
    for item_line in df_cm.iterrows():
        sum_lin.append(item_line[1].sum())
    df_cm['Recall'] = sum_lin
    sum_col.append(np.sum(sum_lin))
    df_cm.loc['Precision'] = sum_col


def pretty_plot_confusion_matrix(df_cm, snr, save_path, annot=True, cmap="Oranges", fmt='.2f', fz=9,
                                 lw=0.5, cbar=False, fig_size=None, show_null_values=0, pred_val_axis='x'):
    """
      print conf matrix with default layout (like matlab)
      params:
        df_cm          dataframe (pandas) without totals
        save_path      path to save confusion map
        annot          print text in each cell
        cmap           Oranges,Oranges_r,YlGnBu,Blues,RdBu, ... see:
        fz             fontsize
        lw             linewidth
        pred_val_axis  where to show the prediction values (x or y axis)
                        'col' or 'x': show predicted values in columns (x axis) instead lines
                        'lin' or 'y': show predicted values in lines   (y axis)
    """
    if fig_size is None:
        fig_size = [8, 8]
    if pred_val_axis in ('col', 'x'):
        xlbl = 'Predicted Modulation'
        ylbl = 'True Modulation'
    else:
        xlbl = 'True Modulation'
        ylbl = 'Predicted Modulation'
        df_cm = df_cm.T
    print('Save: ' + save_path)
    # create "Total" column
    insert_totals(df_cm)

    # this is for print always in the same window
    if snr is None:
        fig, ax1 = get_new_fig(
            'Confusion Matrix of All SNRs', fig_size)
    else:
        fig, ax1 = get_new_fig('Confusion Matrix of %ddB' % snr, fig_size)

    # thanks for seaborn
    ax = sn.heatmap(df_cm, annot=annot, annot_kws={"size": fz}, linewidths=lw, ax=ax1,
                    cbar=cbar, cmap=cmap, linecolor='black', fmt=fmt)

    # set tick labels rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=25, fontsize=10)

    # Turn off all the ticks
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # face colors list
    quadmesh = ax.findobj(QuadMesh)[0]
    face_colors = quadmesh.get_facecolors()

    # iter in text elements
    array_df = np.array(df_cm.to_records(index=False).tolist())
    text_add = []
    text_del = []
    position = -1  # from left to right, bottom to top.
    for t in ax.collections[0].axes.texts:  # ax.texts:
        pos = np.array(t.get_position()) - [0.5, 0.5]
        lin = int(pos[1])
        col = int(pos[0])
        position += 1
        # print ('>>> pos: %s, position: %s, val: %s, txt: %s' %(pos, position, array_df[lin][col], t.get_text()))

        # set text
        txt_res = config_cell_text_and_colors(
            array_df, lin, col, t, face_colors, position, fz, show_null_values)

        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])

    # remove the old ones
    for item in text_del:
        item.remove()
    # append the new ones
    for item in text_add:
        ax.text(item['x'], item['y'], item['text'], **item['kw'])

    # titles and legends
    if snr is None:
        ax.set_title('Confusion Matrix of All SNRs',
                     fontsize=18, fontweight='bold')
    else:
        ax.set_title('Confusion Matrix of %ddB' %
                     snr, fontsize=18, fontweight='bold')
    ax.set_xlabel(xlbl, fontsize=18, fontweight='bold')
    ax.set_ylabel(ylbl, fontsize=18, fontweight='bold')
    ax.tick_params(which='major', bottom=True,
                   top=False, left=True, right=False)
    plt.tight_layout()  # set layout slim
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


@CONFUSIONS.register_module()
class ConfusionMap(object):
    def __init__(self, log_dir, name, method, only_all=False):
        self.log_dir = log_dir
        self.name = name
        self.config = method['config']
        if 'has_snr_classifier' in method:
            self.has_snr_classifier = method['has_snr_classifier']
        else:
            self.has_snr_classifier = False
        self.only_all = only_all
        self.format_out_dir = os.path.join(
            self.log_dir, self.config, 'format_out')
        if self.has_snr_classifier:
            # self.snr_results = np.load(os.path.join(
            #     self.format_out_dir, 'snr_pre.npy'))
            self.low_results = np.load(os.path.join(
                self.format_out_dir, 'low_pre.npy'))
            self.high_results = np.load(os.path.join(
                self.format_out_dir, 'high_pre.npy'))
            self.merge_results = np.load(os.path.join(
                self.format_out_dir, 'merge_pre.npy'))
        else:
            self.results = np.load(os.path.join(
                self.format_out_dir, 'pre.npy'))

        self.SNRS, self.CLASSES, self.mods_dict, self.snrs_dict, self.ann_info = load_annotation(
            os.path.join(self.format_out_dir, 'ann.json'))

    def _evaluate(self, results, save_dir, prefix=None):
        confusion_matrix = np.zeros((len(self.SNRS), len(
            self.CLASSES), len(self.CLASSES)), dtype=np.float64)
        fig_size = [int(len(self.CLASSES) / 11 * 7), int(len(self.CLASSES) / 11 * 7)]
        for idx in range(len(self.ann_info)):
            ann = self.ann_info[idx]
            snrs = ann['snrs']
            labels = ann['mod_labels']
            if len(snrs) == 1 and len(labels) == 1:
                predict_class_index = int(np.argmax(results[idx, :]))
                confusion_matrix[self.snrs_dict['{:.3f}'.format(snrs[0])],
                                 labels[0], predict_class_index] += 1
            else:
                raise ValueError('Please check your dataset, the size of snrs and labels are both 1 for any item. '
                                 'However, the current item with the idx {:d} has the snrs size {:d} and the '
                                 'labels size {:d}'.format(idx, snrs.size, labels.size))

        if not self.only_all:
            for snr_index, snr in enumerate(self.SNRS):
                conf = confusion_matrix[snr_index, :, :]
                if prefix is None:
                    save_path = os.path.join(
                        save_dir, 'snr_' + str(int(snr)) + '_' + self.name)
                else:
                    save_path = os.path.join(
                        save_dir, prefix + '_' + 'snr_' + str(int(snr)) + '_' + self.name)

                # get pandas dataframe
                df_cm = DataFrame(conf, index=self.CLASSES,
                                  columns=self.CLASSES)
                # colormap: see this and choose your more dear
                pretty_plot_confusion_matrix(df_cm, snr, save_path, fig_size=fig_size)

        conf = np.sum(confusion_matrix, axis=0)
        if prefix is None:
            save_path = os.path.join(save_dir, self.name)
        else:
            save_path = os.path.join(save_dir, prefix + '_' + self.name)

        # get pandas dataframe
        df_cm = DataFrame(conf, index=self.CLASSES, columns=self.CLASSES)
        # colormap: see this and choose your more dear
        pretty_plot_confusion_matrix(df_cm, None, save_path, fig_size=fig_size)

    def plot(self, save_dir):
        if self.has_snr_classifier:
            # self._evaluate(self.snr_results, 'snr')
            only_all = self.only_all

            self.only_all = True
            self._evaluate(self.low_results, save_dir, 'low')
            self._evaluate(self.high_results, save_dir, 'high')

            self.only_all = only_all
            self._evaluate(self.merge_results, save_dir, 'merge')
        else:
            self._evaluate(self.results, save_dir)


if __name__ == '__main__':
    confusion = ConfusionMap('/home/citybuster/Data/SignalProcessing/Workdir',
                             'confusion_map_mldnn.pdf', 'mldnn_mlnetv5_640_0.0004_0.5_deepsig_201610A', True)
    confusion.run(
        '/home/citybuster/Data/SignalProcessing/Workdir/mldnn_mlnetv5_640_0.0004_0.5_deepsig_201610A/fig')
