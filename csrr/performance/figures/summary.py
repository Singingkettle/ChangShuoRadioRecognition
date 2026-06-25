#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""Modulation summary markdown table.

Produces a markdown file containing:
  - one accuracy-vs-SNR table per dataset (rows: methods, columns: SNRs + MAA)
  - one F1-per-class table per dataset (rows: methods, columns: classes + MAF)

The previous implementation referenced ``.f1_score`` and iterated ``.ACC`` as
a list. ``ClassificationMetricsWithSNRForSingle`` actually exposes ``.ACC`` and
``.F1`` as dicts keyed by ``"<snr>dB"`` plus ``"All SNRs"``, with ``.F1[key]``
itself being a dict keyed by class name (plus ``"Mean"``). This file aligns the
table generator with that contract.
"""

import os

from ..builder import TABLES


def _format_snr_header(snr_key):
    """Render an SNR column header.

    ``snr_key`` is something like ``'12dB'`` or ``'-2dB'``; we keep it as-is
    instead of trying to coerce it to an int (which fails for negative or
    non-integer values).
    """
    return str(snr_key)


@TABLES.register_module()
class ModulationSummary:
    def __init__(self,
                 dataset,
                 legend=None,
                 scatter=None,
                 legend_config=None,
                 scatter_config=None):
        self.dataset = dataset
        # Accept both ``legend``/``scatter`` (new builder convention) and the
        # legacy ``legend_config``/``scatter_config`` arg names.
        self.legend = legend if legend is not None else legend_config
        self.scatter = scatter if scatter is not None else scatter_config

    def __call__(self, performances, save_dir):
        content = '# Summary of all Algorithms  \n'
        for dataset_name in self.dataset:
            if dataset_name not in performances:
                continue
            content += f'## Experimental results of dataset {dataset_name}  \n'

            # ---- Accuracy vs SNR ----
            content += f'### SNR Accuracy Table  \n'
            content += self._render_accuracy_table(
                self.dataset[dataset_name], performances[dataset_name])

            # ---- F1 per class ----
            content += f'### Modulation F1 Score Table  \n'
            content += self._render_f1_table(
                self.dataset[dataset_name], performances[dataset_name])

        save_path = os.path.join(save_dir, 'summary.md')
        with open(save_path, 'w') as f:
            f.write(content)
        print(f'Save: {save_path}')

    @staticmethod
    def _render_accuracy_table(method_names, dataset_performances):
        # Build SNR header from the union of available SNR keys
        # (skip the 'All SNRs' summary entry; it goes in the MAA column).
        all_snr_keys = []
        for method_name in method_names:
            if method_name not in dataset_performances:
                continue
            acc = dataset_performances[method_name].ACC
            for k in acc:
                if k == 'All SNRs':
                    continue
                if k not in all_snr_keys:
                    all_snr_keys.append(k)

        if not all_snr_keys:
            return '_No data available for this dataset._  \n'

        header = '| Method '
        sep = '|:---:'
        for snr_key in all_snr_keys:
            header += '| ' + _format_snr_header(snr_key) + ' '
            sep += '|:---:'
        header += '| MAA |  \n'
        sep += '|:---:|  \n'
        table = header + sep

        for method_name in method_names:
            if method_name not in dataset_performances:
                continue
            acc = dataset_performances[method_name].ACC
            line = '| ' + str(method_name) + ' '
            for snr_key in all_snr_keys:
                v = acc.get(snr_key)
                line += '| {:.3f} '.format(v) if v is not None else '| - '
            line += '| {:.3f} |  \n'.format(acc.get('All SNRs', float('nan')))
            table += line

        return table

    @staticmethod
    def _render_f1_table(method_names, dataset_performances):
        # Pick the class list from the first available method
        classes = None
        for method_name in method_names:
            if method_name in dataset_performances:
                classes = list(dataset_performances[method_name].classes)
                break

        if classes is None:
            return '_No data available for this dataset._  \n'

        header = '| Method '
        sep = '|:---:'
        for class_name in classes:
            header += '| ' + str(class_name) + ' '
            sep += '|:---:'
        header += '| MAF |  \n'
        sep += '|:---:|  \n'
        table = header + sep

        for method_name in method_names:
            if method_name not in dataset_performances:
                continue
            f1 = dataset_performances[method_name].F1
            # F1 is dict[snr_key -> dict[class_name -> float]] + 'All SNRs'.
            # We use the 'All SNRs' aggregate.
            f1_all = f1.get('All SNRs')
            if f1_all is None:
                # Fall back to first available SNR bucket
                first_key = next(iter(f1.keys()))
                f1_all = f1[first_key]
            line = '| ' + str(method_name) + ' '
            for class_name in classes:
                v = f1_all.get(class_name)
                line += '| {:.3f} '.format(v) if v is not None else '| - '
            line += '| {:.3f} |  \n'.format(f1_all.get('Mean', float('nan')))
            table += line

        return table
