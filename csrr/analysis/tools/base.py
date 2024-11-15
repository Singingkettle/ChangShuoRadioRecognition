from abc import ABCMeta, abstractmethod

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from matplotlib.collections import QuadMesh
from pandas import DataFrame
from sklearn.metrics import silhouette_score, silhouette_samples

from .utils import get_new_fig, radar_factory
from ..builder import FIGURES

mpl.rc('font', family='Times New Roman')
plt.rcParams["font.family"] = "Times New Roman"


def config_cell_text_and_colors(array_df, lin, col, o_text, face_colors, position, fz, show_null_values=0):
    """
      figure_configs cell text and colors
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


def pretty_plot_confusion_matrix(df_cm, snr, save_path, annot=True, cmap='Oranges', fmt='.2f', fz=9,
                                 lw=0.5, cbar=False, fig_size=None, show_null_values=0, pred_val_axis='x'):
    """
      print conf matrix with default layout (like matlab)
      params:
        df_cm          dataframe (pandas) without totals
        save_path      path to format confusion map
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

    fig, ax1 = get_new_fig(f'Confusion Matrix of {snr}', fig_size)

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

    x = np.arange(0, 1 + len(df_cm.index), 1)
    y = np.arange(0, 1 + len(df_cm.index), 1)
    face_colors = np.reshape(face_colors, [len(df_cm.index), len(df_cm.index), 4])
    ax.pcolormesh(x, y, face_colors, edgecolors='black')
    # remove the old ones
    for item in text_del:
        item.remove()
    # append the new ones
    for item in text_add:
        ax.text(item['x'], item['y'], item['text'], **item['kw'])

    # titles and legends
    ax.set_title(f'Confusion Matrix of {snr}', fontsize=24, fontweight='bold')
    ax.set_xlabel(xlbl, fontsize=24, fontweight='bold')
    ax.set_ylabel(ylbl, fontsize=24, fontweight='bold')
    ax.tick_params(which='major', bottom=True,
                   top=False, left=True, right=False)
    plt.tight_layout()  # set layout slim
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


@FIGURES.register_module()
class BaseDraw(metaclass=ABCMeta):
    def __init__(self, dataset, plot_config=None):
        self.dataset = dataset
        if plot_config is None:
            self.plot_config = dict(loc='lower right', prop={'size': 16, 'weight': 'bold'})
        else:
            self.plot_config = plot_config
        self.xticklabel_rotation = 50

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def _draw_fea_distribution(self, classes, feas, gts, centers, save_path, method_name):
        print('Save: ' + save_path)
        num_class = len(classes)

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        ax1.set_xlim([-1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(feas) + (num_class + 1) * 10])

        silhouette_avg = silhouette_score(feas, gts)
        sample_silhouette_values = silhouette_samples(feas, gts)

        y_lower = 10
        for class_index in range(num_class):
            ith_cluster_silhouette_values = sample_silhouette_values[gts == class_index]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(class_index) / num_class)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
                label=f'{class_index:d}-{classes[class_index]}'
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(class_index))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various modulations.", fontsize=24, fontweight="bold")
        ax1.set_xlabel("The silhouette coefficient values", fontsize=24, fontweight="bold")
        ax1.set_ylabel("Modulation label", fontsize=24, fontweight="bold")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([float(f'{v / 10:.1f}') for v in np.arange(-10, 11, 1)])
        ax1.tick_params(axis='both', which='major', labelsize=14)

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(gts.astype(float) / num_class)
        ax2.scatter(
            feas[:, 0], feas[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k",
        )

        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for class_index in range(num_class):
            color = cm.nipy_spectral(float(class_index) / num_class)
            color = np.array([color])
            c = centers[class_index, :]
            ax2.scatter(c[0], c[1], marker="$%d$" % class_index, alpha=1, s=50, edgecolor="k",
                        c=color, label=f'{class_index:d}-{classes[class_index]}')

        ax2.set_title("The visualization of the feature data.", fontsize=24, fontweight="bold")
        ax2.set_xlabel("Feature space for the 1st feature", fontsize=24, fontweight="bold")
        ax2.set_ylabel("Feature space for the 2nd feature", fontsize=24, fontweight="bold")
        ax2.tick_params(axis='both', which='major', labelsize=14)
        plt.suptitle(
            f'Silhouette analysis for {method_name} at 12dB on RadioML.2016.10A, '
            f'and Average Silhouette Score = {silhouette_avg:.2f}',
            fontsize=24,
            fontweight="bold",
        )

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05),
                   prop={'size': 14, 'weight': 'bold'}, ncol=num_class, edgecolor='black')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

    def _draw_plot(self, methods, legend, xs, x_label, y_label, title, save_path):
        print('Save: ' + save_path)
        fig, ax = get_new_fig('Curve', [10, 10])
        xs_str = ['%9s' % i for i in xs]
        xs = np.array([i for i in range(len(xs))]) / (len(xs) - 1)
        ax.set_xticks(xs)  # values
        ax.set_xticklabels(xs_str)  # labels

        ys = np.array([i for i in range(11)]) / 10
        ys_str = ['%.1f' % i for i in ys]
        ax.set_yticks(ys)  # values
        ax.set_yticklabels(ys_str)  # labels
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        for method in methods:
            name = method['name']
            point = method['point']
            score = method['score']

            ys = np.array(point)
            legend_name = name + ' [{:.4f}]'.format(score)

            ax.plot(
                xs, ys, label=legend_name, linewidth=1,
                color=legend[name]['color'],
                linestyle=legend[name]['linestyle'],
                marker=legend[name]['marker'],
                markersize=6,
            )

        leg = ax.legend(**self.plot_config)

        leg.get_frame().set_edgecolor('black')
        plt.setp(leg.texts, family='Times New Roman')

        ax.set_xlabel(x_label, fontsize=24, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=24, fontweight='bold')
        ax.set_title(title, fontsize=24, fontweight='bold')

        # Don't allow the axis to be on top of your data
        ax.set_axisbelow(True)

        # Turn on the minor TICKS, which are required for the minor GRID
        ax.minorticks_on()

        # Customize the major grid
        ax.grid(visible=True, which='major', linestyle='-',
                linewidth='0.5', color='black', alpha=0.2)
        # # Customize the minor grid
        # ax.grid(b=True, which='minor', linestyle=':',
        #         linewidth='0.5', color='black', alpha=0.5)

        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.tick_params(which='minor', bottom=False,
                        top=False, left=False, right=False)
        plt.tick_params(which='major', bottom=True,
                        top=False, left=True, right=False)

        ax.tick_params(which='minor', bottom=False,
                       top=False, left=False, right=False)
        ax.tick_params(which='major', bottom=True,
                       top=False, left=True, right=False)
        plt.setp(ax.get_xticklabels(), rotation=self.xticklabel_rotation,
                 horizontalalignment='right')
        plt.tight_layout()  # set layout slim
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

    def _draw_radar(self, methods, scatter, xs, title, save_path):
        print('Save: ' + save_path)
        theta = radar_factory(len(xs), frame='polygon')
        fig, ax = plt.subplots(figsize=(8, 8), nrows=1, ncols=1, subplot_kw=dict(projection='radar'))
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)

        f1s_list = []
        for method in methods:
            name = method['name']
            point = method['point']
            score = method['score']
            legend_name = name + ' [{:.3f}]'.format(score)
            f1s_list.append(score)
            ax.plot(theta, point, label=legend_name, color=scatter[name]['color'], linewidth=0.1)
            ax.fill(theta, point, facecolor=scatter[name]['color'], alpha=0.25)
        ax.set_rgrids([0.1, 0.3, 0.5, 0.7, 0.9], angle=-45, fontsize=18)
        ax.set_thetagrids(np.degrees(theta), xs, fontsize=18)
        leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                        prop={'size': 18, 'weight': 'bold'}, handletextpad=0.2,
                        markerscale=20, ncol=2, columnspacing=0.2)
        leg.get_frame().set_edgecolor('black')
        plt.setp(leg.texts, family='Times New Roman')
        ax.set_title(title, fontsize=24, fontweight='bold')
        plt.tight_layout()  # set layout slim
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

    def _draw_confusion_map(self, conf, snr, save_path, classes):

        fig_size = [int(len(classes) / 11 * 7), int(len(classes) / 11 * 7)]
        # get pandas dataframe
        df_cm = DataFrame(conf, index=classes, columns=classes)
        # colormap: see this and choose your more dear
        pretty_plot_confusion_matrix(df_cm, snr, save_path, fig_size=fig_size)

    def _draw_train(self, methods, legend, title, save_path):
        pass
