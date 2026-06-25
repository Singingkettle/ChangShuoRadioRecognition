"""Training-curve plot.

Produces two overlay plots per dataset:
  * train_loss_<dataset>.pdf      training loss vs epoch (last iter of each)
  * val_accuracy_<dataset>.pdf    validation accuracy vs epoch

The mmengine work directory layout is::

    <work_dir>/<publish_subdir>/<timestamp>/vis_data/<timestamp>.json   (optional)
    <work_dir>/<publish_subdir>/<timestamp>/<timestamp>.log             (fallback)

We accept either; missing runs are skipped with a warning.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from .base import BaseDraw
from .utils import load_run_logs
from ..builder import FIGURES

plt.rcParams["font.family"] = "Times New Roman"


def _series_from_log(log_dict, metric, mode_filter=None):
    """Return ``(epochs, values)`` arrays for ``metric`` extracted from log_dict.

    For each epoch we take the *last* recorded value of ``metric`` (i.e. the
    end-of-epoch summary). If ``mode_filter`` is set (``'train'`` or
    ``'val'``), we only consider iterations recorded for that mode.
    """
    epochs = sorted(log_dict.keys())
    xs, ys = [], []
    for epoch in epochs:
        bucket = log_dict[epoch]
        if metric not in bucket:
            continue
        values = bucket[metric]
        if mode_filter is not None and 'mode' in bucket:
            filtered = [v for v, m in zip(values, bucket['mode'])
                        if m == mode_filter]
            if filtered:
                values = filtered
            else:
                continue
        xs.append(epoch)
        ys.append(values[-1])
    return np.array(xs), np.array(ys)


def _plot_curve(curves, title, xlabel, ylabel, save_path, legend_config=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, (xs, ys, style) in curves.items():
        if len(xs) == 0:
            continue
        kwargs = dict(label=label, linewidth=1.0)
        if style is not None:
            for k in ('color', 'linestyle', 'marker'):
                if k in style and style[k] is not None:
                    kwargs[k] = style[k]
        ax.plot(xs, ys, **kwargs)
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(visible=True, which='major', linestyle='-', linewidth='0.5',
            color='black', alpha=0.2)
    ax.set_axisbelow(True)
    if curves:
        leg = ax.legend(loc='best', prop={'size': 10, 'weight': 'bold'})
        leg.get_frame().set_edgecolor('black')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Save: {save_path}')


@FIGURES.register_module()
class TrainPlot(BaseDraw):
    """Per-dataset training curves overlaid across methods.

    Args:
        dataset (Dict[str, List[str]]): ``{dataset_name: [method_name, ...]}``.
        loss_metric (str): mmengine metric name for training loss. Default
            ``'loss'`` (works for single-loss models; multi-loss models also
            write a summed ``loss`` field).
        val_metric (str): mmengine metric name for validation accuracy. Default
            ``'accuracy/top1'``.
        max_epochs (int): Truncate plotted x-axis to this many epochs.
            ``None`` keeps all epochs (default).
        legend (Dict[str, dict]): Per-method legend style ``{name: {color,
            linestyle, marker}}``, typically provided by ``Classification``.
        scatter (Any): Unused; accepted for builder symmetry.
    """

    def __init__(self,
                 dataset,
                 loss_metric='loss',
                 val_metric='accuracy/top1',
                 max_epochs=None,
                 legend=None,
                 scatter=None,
                 plot_config=None):
        super().__init__(dataset, plot_config)
        self.loss_metric = loss_metric
        self.val_metric = val_metric
        self.max_epochs = max_epochs
        self.legend = legend or {}
        self.scatter = scatter

    def __call__(self, performances, save_dir):  # noqa: ARG002
        os.makedirs(save_dir, exist_ok=True)
        publish = self._resolve_publish(performances)
        work_dir = self._resolve_work_dir(performances)

        for dataset_name, method_names in self.dataset.items():
            loss_curves = {}
            acc_curves = {}
            for method_name in method_names:
                cfg_subdir = publish.get(dataset_name, {}).get(method_name)
                if cfg_subdir is None:
                    print(f'[TrainPlot] No publish entry for {method_name}'
                          f'/{dataset_name}; skipping.')
                    continue
                run_dir = os.path.join(work_dir, cfg_subdir)
                log_dict = load_run_logs(run_dir)
                if not log_dict:
                    print(f'[TrainPlot] No logs found under {run_dir}; '
                          f'skipping {method_name}.')
                    continue

                style = self.legend.get(method_name)
                xs, ys = _series_from_log(log_dict, self.loss_metric,
                                          mode_filter='train')
                if self.max_epochs is not None:
                    mask = xs <= self.max_epochs
                    xs, ys = xs[mask], ys[mask]
                loss_curves[method_name] = (xs, ys, style)

                xs, ys = _series_from_log(log_dict, self.val_metric,
                                          mode_filter='val')
                # mmengine logs accuracy/top1 as percent; normalize to 0-1 for
                # nicer plots if values look like percentages.
                if len(ys) > 0 and ys.max() > 1.5:
                    ys = ys / 100.0
                if self.max_epochs is not None:
                    mask = xs <= self.max_epochs
                    xs, ys = xs[mask], ys[mask]
                acc_curves[method_name] = (xs, ys, style)

            if loss_curves:
                save_path = os.path.join(
                    save_dir, f'train_loss_{dataset_name}.pdf')
                _plot_curve(loss_curves,
                            f'Training Loss ({dataset_name})',
                            'Epoch', 'Loss', save_path)
            if acc_curves:
                save_path = os.path.join(
                    save_dir, f'val_accuracy_{dataset_name}.pdf')
                _plot_curve(acc_curves,
                            f'Validation Accuracy ({dataset_name})',
                            'Epoch', 'Accuracy', save_path)

    @staticmethod
    def _resolve_publish(performances):
        if isinstance(performances, dict) and '_info' in performances:
            return performances['_info'].get('publish', {})
        return {}

    @staticmethod
    def _resolve_work_dir(performances):
        if isinstance(performances, dict) and '_info' in performances:
            return performances['_info'].get('work_dir', 'work_dirs')
        return 'work_dirs'


# Backward-compat alias for older configs that reference ``LossAccuracyPlot``.
@FIGURES.register_module()
class LossAccuracyPlot(TrainPlot):
    pass
