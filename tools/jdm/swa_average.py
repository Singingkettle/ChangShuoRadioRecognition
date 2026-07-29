"""Average detector snapshot checkpoints (SWA) into a single checkpoint.

Usage:
    python tools/jdm/swa_average.py --work-dir work_dirs/jdm/retune/det_swa_from120_w22 \
        --first-epoch 4 --out work_dirs/jdm/retune/det_swa_from120_w22/swa_avg.pth
"""
import argparse
import glob
import os
import re

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work-dir', required=True)
    ap.add_argument('--first-epoch', type=int, default=4,
                    help='skip the first N-1 adaptation epochs')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    paths = []
    for p in glob.glob(os.path.join(args.work_dir, 'epoch_*.pth')):
        m = re.search(r'epoch_(\d+)\.pth$', p)
        if m and int(m.group(1)) >= args.first_epoch:
            paths.append((int(m.group(1)), p))
    paths.sort()
    if not paths:
        raise SystemExit(f'no epoch_*.pth >= {args.first_epoch} in {args.work_dir}')
    print(f'[swa] averaging {len(paths)} snapshots: '
          f'epochs {paths[0][0]}..{paths[-1][0]}')

    avg = None
    n = 0
    template = None
    for _, p in paths:
        ckpt = torch.load(p, map_location='cpu')
        sd = ckpt.get('state_dict', ckpt)
        if avg is None:
            template = ckpt
            avg = {k: v.double().clone() if torch.is_floating_point(v) else v
                   for k, v in sd.items()}
        else:
            for k, v in sd.items():
                if torch.is_floating_point(v):
                    avg[k] += v.double()
        n += 1
    for k in avg:
        if torch.is_floating_point(avg[k]):
            avg[k] = (avg[k] / n).to(template['state_dict'][k].dtype
                                     if 'state_dict' in template else
                                     template[k].dtype)
    out = {'state_dict': avg, 'meta': {'swa_of': [p for _, p in paths]}}
    torch.save(out, args.out)
    print(f'[swa] wrote {args.out} ({n} snapshots averaged)')


if __name__ == '__main__':
    main()
