#!/usr/bin/env python3
"""Generate current RCPS evidence figures from landed CSV summaries."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path('/home/citybuster/Data/RCPS/work_dirs/paper_evidence/figures')
OUT_DIR.mkdir(parents=True, exist_ok=True)
CB = {
    'hard-ce': '#0072B2',
    'static-ls': '#E69F00',
    'rcps-retention-eps0p10': '#009E73',
    'RCPS': '#009E73',
    'Static LS': '#E69F00',
}


def save(fig, name: str) -> None:
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(OUT_DIR / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_cifar10c_resnet18_severity() -> None:
    agg = Path('/home/citybuster/Data/RCPS/work_dirs/crossmodal_vision_cifar10c_30ep/summary/cifar10c_resnet18_hard_static_rcps_eps0p10_3seed_aggregate.csv')
    df = pd.read_csv(agg)
    df = df[(df['group'] == 'severity') & (df['method_variant'].isin(['hard-ce', 'static-ls', 'rcps-retention-eps0p10']))].copy()
    df['severity'] = df['reliability_bin'].astype(int)
    metrics = [('accuracy_mean', 'Accuracy (%)'), ('nll_mean', 'NLL'), ('ece_mean', 'ECE'), ('brier_mean', 'Brier')]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True)
    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        for method in ['hard-ce', 'static-ls', 'rcps-retention-eps0p10']:
            sub = df[df['method_variant'] == method].sort_values('severity')
            ax.plot(sub['severity'], sub[metric], marker='o', linewidth=1.8, markersize=4,
                    color=CB[method], label={'hard-ce': 'Hard CE', 'static-ls': 'Static LS', 'rcps-retention-eps0p10': 'RCPS'}[method])
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, linewidth=0.6)
    for ax in axes[-1]:
        ax.set_xlabel('Corruption severity')
    axes[0, 0].legend(frameon=False, fontsize=8, loc='best')
    fig.suptitle('CIFAR-10-C / ResNet18-CIFAR: reliability-stratified behavior', fontsize=10)
    save(fig, 'fig_cifar10c_resnet18_severity')


def plot_cifar10c_resnet18_deltas() -> None:
    delta = Path('/home/citybuster/Data/RCPS/work_dirs/crossmodal_vision_cifar10c_30ep/summary/cifar10c_resnet18_static_rcps_eps0p10_vs_hard_3seed_delta_aggregate.csv')
    df = pd.read_csv(delta)
    df = df[(df['group'] == 'severity') & (df['method_variant'].isin(['static-ls', 'rcps-retention-eps0p10']))].copy()
    df['severity'] = df['reliability_bin'].astype(int)
    metrics = [('delta_accuracy_mean', 'Delta Accuracy (pp)'), ('delta_nll_mean', 'Delta NLL'), ('delta_ece_mean', 'Delta ECE'), ('delta_brier_mean', 'Delta Brier')]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True)
    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        ax.axhline(0, color='0.25', linewidth=0.8)
        for method, label in [('static-ls', 'Static LS'), ('rcps-retention-eps0p10', 'RCPS')]:
            sub = df[df['method_variant'] == method].sort_values('severity')
            ax.plot(sub['severity'], sub[metric], marker='o', linewidth=1.8, markersize=4,
                    color=CB[label], label=label)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, linewidth=0.6)
    for ax in axes[-1]:
        ax.set_xlabel('Corruption severity')
    axes[0, 0].legend(frameon=False, fontsize=8, loc='best')
    fig.suptitle('CIFAR-10-C / ResNet18-CIFAR: deltas versus Hard CE', fontsize=10)
    save(fig, 'fig_cifar10c_resnet18_delta')


def plot_evidence_bar() -> None:
    csv_path = Path('/home/citybuster/Data/RCPS/work_dirs/paper_evidence/rcps_current_evidence_summary.csv')
    df = pd.read_csv(csv_path)
    keep = df[df['tier'].str.contains('main|supplementary', regex=True)].copy()
    labels = keep['modality'] + '\n' + keep['dataset'] + '\n' + keep['model']
    x = np.arange(len(keep))
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    vals = keep['delta_accuracy_pp'].astype(float)
    colors = ['#009E73' if v >= 0 else '#D55E00' for v in vals]
    ax.bar(x, vals, color=colors, width=0.65)
    ax.axhline(0, color='0.25', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=7)
    ax.set_ylabel('Delta accuracy vs Hard CE (pp)')
    ax.grid(True, axis='y', alpha=0.25, linewidth=0.6)
    ax.set_title('Current RCPS positive evidence summary', fontsize=10)
    save(fig, 'fig_current_evidence_accuracy_delta')


def main() -> None:
    plot_cifar10c_resnet18_severity()
    plot_cifar10c_resnet18_deltas()
    plot_evidence_bar()
    print(f'Wrote figures to {OUT_DIR}')


if __name__ == '__main__':
    main()
