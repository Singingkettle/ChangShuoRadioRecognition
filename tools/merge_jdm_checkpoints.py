# Copyright (c) Shuo Chang. All Rights Reserved.
"""Merge the separately trained JDM module checkpoints into one state dict
usable by ``configs/jdm/jdm-joint_iq-csrd.py``:

    python tools/merge_jdm_checkpoints.py \
        work_dirs/jdm-det_fft-csrd/best_detection_mAP_epoch_X.pth \
        work_dirs/jdm-amc_iq-csrd/best_accuracy_top1_epoch_Y.pth \
        work_dirs/jdm-joint_iq-csrd/jdm_joint.pth
"""
import argparse

import torch


def main():
    parser = argparse.ArgumentParser(
        description='Merge JDM detector + classifier checkpoints')
    parser.add_argument('det_checkpoint', help='SignalDetector checkpoint')
    parser.add_argument('amc_checkpoint', help='SignalClassifier checkpoint')
    parser.add_argument('out', help='output joint checkpoint path')
    args = parser.parse_args()

    merged = {}
    for prefix, path in (('detector', args.det_checkpoint),
                         ('classifier', args.amc_checkpoint)):
        ckpt = torch.load(path, map_location='cpu')
        state = ckpt.get('state_dict', ckpt)
        merged.update({f'{prefix}.{k}': v for k, v in state.items()})

    torch.save(dict(state_dict=merged), args.out)
    print(f'saved {len(merged)} parameters/buffers to {args.out}')


if __name__ == '__main__':
    main()
