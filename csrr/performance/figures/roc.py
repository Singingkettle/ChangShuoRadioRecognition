"""ROC curve figure.

Consumes ``performance.roc`` which is structured as::

    roc['ovr'][<snr_or_'micro'>] -> {fpr, tpr, auc}
        where each inner dict is keyed by class index and the special keys
        ``'micro'`` and ``'macro'``.

For comparison plots we typically want one curve per method (macro-averaged)
for each requested SNR group.
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
class ROCCurve(BaseDraw):
    """One ROC-curve PDF per (dataset, SNR group).

    Args:
        dataset (Dict[str, Dict[str, List[str]]]): mapping of the form
            ``{dataset_name: {group_name: [method, ...]}}``. ``group_name`` is
            just a label used in the saved file name; one PDF is produced per
            group.
        snr_groups (List[Any]): SNR keys to render (one PDF per key per
            group). Defaults to ``['micro']`` which uses the aggregated curve
            across all SNRs.
        average (str): ``'macro'`` (default) or ``'micro'`` -- which averaged
            curve to draw per method.
        legend (Dict[str, dict]): Per-method legend style.
        scatter (Any): Unused; accepted for builder symmetry.
    """

    def __init__(self,
                 dataset,
                 snr_groups=None,
                 average='macro',
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
                    self._plot_group(
                        dataset_name, group_name, method_names, snr_key,
                        performances[dataset_name], save_dir)

    def _plot_group(self, dataset_name, group_name, method_names, snr_key,
                    method_perfs, save_dir):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot([0, 1], [0, 1], linestyle=':', color='gray', linewidth=1,
                label='Chance')

        any_curve = False
        for method_name in method_names:
            if method_name not in method_perfs:
                print(f'[ROCCurve] {method_name} missing for {dataset_name}; skipping.')
                continue
            roc = method_perfs[method_name].roc
            ovr = roc.get('ovr', roc)
            entry = ovr.get(snr_key)
            if entry is None:
                for k in ovr.keys():
                    if str(k) == str(snr_key):
                        entry = ovr[k]
                        break
            if entry is None:
                print(f'[ROCCurve] {method_name}: no ROC for snr={snr_key}; skipping.')
                continue
            fpr_dict = entry.get('fpr', {})
            tpr_dict = entry.get('tpr', {})
            auc_dict = entry.get('auc', {})
            fpr = fpr_dict.get(self.average)
            tpr = tpr_dict.get(self.average)
            auc_value = auc_dict.get(self.average)
            if fpr is None or tpr is None:
                continue
            style = self.legend.get(method_name, {})
            label = '{} (AUC={:.3f})'.format(
                method_name,
                auc_value if auc_value is not None else float('nan'))
            ax.plot(np.asarray(fpr), np.asarray(tpr), label=label,
                    linewidth=1.2,
                    color=style.get('color'),
                    linestyle=style.get('linestyle', '-'))
            any_curve = True

        if not any_curve:
            plt.close(fig)
            return

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
        title = (f'ROC Curve ({self.average}) - {dataset_name} @ '
                 f'{_format_snr_key(snr_key)}')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(visible=True, which='major', linestyle='-', linewidth='0.5',
                color='black', alpha=0.2)
        ax.set_axisbelow(True)
        leg = ax.legend(loc='lower right', prop={'size': 10, 'weight': 'bold'})
        leg.get_frame().set_edgecolor('black')
        plt.tight_layout()
        save_path = os.path.join(
            save_dir,
            f'ROC_{group_name}_{dataset_name}_{_format_snr_key(snr_key)}.pdf')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f'Save: {save_path}')
