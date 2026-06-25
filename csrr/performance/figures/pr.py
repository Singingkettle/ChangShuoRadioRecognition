"""Precision-recall curve figure.

Consumes ``performance.precision_recall`` which is structured as::

    pr[<snr_or_'micro'>] -> {precision, recall, average_precision}
        where each inner dict is keyed by class index plus the special key
        ``'micro'``.

For comparison plots we draw one curve per method using the micro-average
within the requested SNR slice.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from .base import BaseDraw
from ..builder import FIGURES

plt.rcParams["font.family"] = "Times New Roman"


def _format_snr_key(snr_key):
    if snr_key in ('micro', 'macro'):
        return snr_key
    try:
        return '{}dB'.format(int(snr_key))
    except (TypeError, ValueError):
        return str(snr_key)


@FIGURES.register_module()
class PRCurve(BaseDraw):
    """One PR-curve PDF per (dataset, SNR group).

    Args:
        dataset (Dict[str, Dict[str, List[str]]]): mapping of the form
            ``{dataset_name: {group_name: [method, ...]}}``.
        snr_groups (List[Any]): SNR keys to render (defaults to ``['micro']``,
            the aggregate across all SNRs).
        average (str): ``'micro'`` (default) or a per-class index.
        legend (Dict[str, dict]): Per-method legend style.
        scatter (Any): Unused; accepted for builder symmetry.
    """

    def __init__(self,
                 dataset,
                 snr_groups=None,
                 average='micro',
                 legend=None,
                 scatter=None,
                 plot_config=None):
        super().__init__(dataset, plot_config)
        self.snr_groups = snr_groups if snr_groups is not None else ['micro']
        self.average = average
        self.legend = legend or {}
        self.scatter = scatter

    def __call__(self, performances, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        for dataset_name, groups in self.dataset.items():
            if dataset_name not in performances:
                continue
            for group_name, method_names in groups.items():
                for snr_key in self.snr_groups:
                    self._plot_group(dataset_name, group_name, method_names,
                                     snr_key, performances[dataset_name],
                                     save_dir)

    def _plot_group(self, dataset_name, group_name, method_names, snr_key,
                    method_perfs, save_dir):
        fig, ax = plt.subplots(figsize=(7, 7))

        any_curve = False
        for method_name in method_names:
            if method_name not in method_perfs:
                print(f'[PRCurve] {method_name} missing for {dataset_name}; skipping.')
                continue
            pr = method_perfs[method_name].precision_recall
            entry = pr.get(snr_key)
            if entry is None:
                for k in pr.keys():
                    if str(k) == str(snr_key):
                        entry = pr[k]
                        break
            if entry is None:
                print(f'[PRCurve] {method_name}: no PR for snr={snr_key}; skipping.')
                continue
            precision = entry.get('precision', {}).get(self.average)
            recall = entry.get('recall', {}).get(self.average)
            ap = entry.get('average_precision', {}).get(self.average)
            if precision is None or recall is None:
                continue
            style = self.legend.get(method_name, {})
            label = '{} (AP={:.3f})'.format(
                method_name, ap if ap is not None else float('nan'))
            ax.plot(np.asarray(recall), np.asarray(precision), label=label,
                    linewidth=1.2,
                    color=style.get('color'),
                    linestyle=style.get('linestyle', '-'))
            any_curve = True

        if not any_curve:
            plt.close(fig)
            return

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
        title = (f'Precision-Recall Curve ({self.average}) - {dataset_name} @ '
                 f'{_format_snr_key(snr_key)}')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(visible=True, which='major', linestyle='-', linewidth='0.5',
                color='black', alpha=0.2)
        ax.set_axisbelow(True)
        leg = ax.legend(loc='lower left', prop={'size': 10, 'weight': 'bold'})
        leg.get_frame().set_edgecolor('black')
        plt.tight_layout()
        save_path = os.path.join(
            save_dir,
            f'PR_{group_name}_{dataset_name}_{_format_snr_key(snr_key)}.pdf')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f'Save: {save_path}')
