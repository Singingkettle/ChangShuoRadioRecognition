# Copyright (c) Shuo Chang. All Rights Reserved.
"""Empirically verify the SNR of a CSRD (twc-profile) dataset export.

For every requested AWGN version the script reconstructs the noise of each
frame and compares the measured SNR against the labeled SNR:

* new exports (with ``wideband_data``): noise = wideband_data - sum(signal_data),
  which also implicitly checks that ``signal_data`` is noise-free;
* old exports (no ``wideband_data``): noise = noisy signal_data - clean
  signal_data of the same item in the ``ideal`` version (v1), and the script
  reports how many sub-signals carry noise (repeated-noise detector).

Example::

    python tools/misc/verify_csrd_snr.py \
        --data-root data/ChangShuoTwc2026 \
        --versions v79 v84 v89 v94 v98
"""
import argparse
import json
import os.path as osp
from glob import glob

import numpy as np
from scipy.io import loadmat


def to_complex(arr):
    """(N, 2, L) real/imag planes -> (N, L) complex."""
    return arr[:, 0, :] + 1j * arr[:, 1, :]


def parse_label_db(snr_label):
    return float('inf') if snr_label == 'infdB' else float(snr_label[:-2])


def verify_version(data_root, version, max_items=None):
    iq_dir = osp.join(data_root, version, 'sequence_data', 'iq')
    anno_dir = osp.join(data_root, version, 'anno')
    files = sorted(glob(osp.join(iq_dir, '*.mat')))
    if max_items:
        files = files[:max_items]

    deltas, noisy_sub_counts, sub_residuals = [], [], []
    for path in files:
        name = osp.splitext(osp.basename(path))[0]
        anno = json.load(open(osp.join(anno_dir, name + '.json')))
        label_db = parse_label_db(anno['snr'][0])
        if not np.isfinite(label_db):
            continue
        mat = loadmat(path)
        subs = to_complex(np.asarray(mat['signal_data'], dtype=np.float64))

        if 'wideband_data' in mat:
            wideband = to_complex(
                np.asarray(mat['wideband_data'], dtype=np.float64))[0]
            noise = wideband - subs.sum(axis=0)
            ref_power = np.mean(np.abs(subs) ** 2, axis=1).mean()
        else:
            clean_path = osp.join(
                data_root, 'v1', 'sequence_data', 'iq', name + '.mat')
            clean = to_complex(np.asarray(
                loadmat(clean_path)['signal_data'], dtype=np.float64))
            per_sub_noise = np.mean(np.abs(subs - clean) ** 2, axis=1)
            noisy_sub_counts.append(int((per_sub_noise > 1e-9).sum()))
            noise = (subs - clean).sum(axis=0)
            ref_power = np.mean(np.abs(clean) ** 2, axis=1).mean()

        noise_power = np.mean(np.abs(noise) ** 2)
        if noise_power <= 0:
            print(f'  {version}/{name}: zero noise power (BUG?)')
            continue
        deltas.append(10 * np.log10(ref_power / noise_power) - label_db)

    deltas = np.asarray(deltas)
    line = (f'{version}: {len(deltas)} noisy frames, '
            f'measured-minus-label SNR: mean {deltas.mean():+.3f} dB, '
            f'std {deltas.std():.3f}, '
            f'range [{deltas.min():+.2f}, {deltas.max():+.2f}]')
    if noisy_sub_counts:
        uniq = dict(zip(*np.unique(noisy_sub_counts, return_counts=True)))
        line += f' | noisy-sub-signal histogram {uniq} (old-style export)'
    print(line)
    return deltas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument(
        '--versions', nargs='+',
        default=['v79', 'v84', 'v89', 'v94', 'v98', 'v105', 'v115', 'v124'],
        help='version folders to verify (default: AWGN + real_awgn spread)')
    parser.add_argument('--max-items', type=int, default=None)
    args = parser.parse_args()

    for version in args.versions:
        verify_version(args.data_root, version, args.max_items)


if __name__ == '__main__':
    main()
