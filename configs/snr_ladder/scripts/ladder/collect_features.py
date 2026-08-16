#!/usr/bin/env python3
"""Extract penultimate features phi (the input to the backbone's final classifier
Linear) together with probs/labels/snrs, for the representation-level analysis.

For these AMC models the classifier lives INSIDE the backbone (e.g. PETCGDNN's
nn.Linear(hidden,num_classes)), so the penultimate phi is the input to that Linear.
We capture it with a forward-pre-hook (the pattern in real_amc_geometry_analysis.py),
NOT via extract_feat(stage='pre_logits') (which would return the logits here).

    python collect_features.py <config> <checkpoint> --split {validation,test} --out feats.pkl
"""
import argparse
import os
import os.path as osp
import pickle
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION

SPLIT_TO_DATALOADER = {'train': 'train_dataloader', 'validation': 'val_dataloader',
                       'val': 'val_dataloader', 'test': 'test_dataloader'}


def set_default_dataloader_cfg(cfg, field):
    if cfg.get(field, None) is None:
        return
    dl = ConfigDict(pin_memory=True, persistent_workers=True,
                    collate_fn=dict(type='default_collate'))
    if digit_version(TORCH_VERSION) < digit_version('1.8.0'):
        dl.persistent_workers = False
    dl.update(deepcopy(cfg[field]))
    cfg[field] = dl


def find_last_linear(module):
    last = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            last = m
    return last


class FeatureHook:
    """Capture the input to the backbone's final Linear classifier = phi."""
    def __init__(self, model):
        self.features = None
        self.source = 'none'
        cand = None
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'classifier'):
            cand = find_last_linear(model.backbone.classifier)
        if cand is None:
            # generic fallback (e.g. MCformer): the final Linear of the whole
            # model is the classification fc; its input is phi.
            cand = find_last_linear(model)
        if cand is not None:
            self.source = 'last_linear_input'
            cand.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        self.features = inputs[0].detach()

    def pop(self):
        f = self.features
        self.features = None
        return f


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config')
    ap.add_argument('checkpoint')
    ap.add_argument('--split', choices=sorted(SPLIT_TO_DATALOADER), default='test')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cfg-options', nargs='+', action=DictAction)
    return ap.parse_args()


def main():
    a = parse_args()
    cfg = Config.fromfile(a.config)
    if a.cfg_options:
        cfg.merge_from_dict(a.cfg_options)
    set_default_dataloader_cfg(cfg, SPLIT_TO_DATALOADER[a.split])
    init_default_scope(cfg.get('default_scope', 'csrr'))

    from csrr.registry import MODELS
    model = MODELS.build(cfg.model)
    load_checkpoint(model, a.checkpoint, map_location='cpu')
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    hook = FeatureHook(model)
    assert hook.source == 'last_linear_input', \
        f'could not find a backbone.classifier Linear to hook (source={hook.source})'

    dataloader = Runner.build_dataloader(cfg[SPLIT_TO_DATALOADER[a.split]])
    dataset = dataloader.dataset

    all_phi, all_pps, all_gts, all_snrs = [], [], [], []
    sc = 0
    print(f'extract phi ({hook.source}) on {a.split}: {len(dataset)} samples', flush=True)
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            results = model.test_step(data)
            phi = hook.pop()
            phi = phi.cpu().numpy() if phi is not None else None
            for j, sample in enumerate(results):
                all_pps.append(sample.pred_score.cpu().numpy())
                all_gts.append(sample.gt_label.item())
                info = dataset.get_data_info(sc)
                all_snrs.append(info.get('snr', sample.get('snr', 0)))
                all_phi.append(phi[j])
                sc += 1
            if (i + 1) % 50 == 0:
                print(f'  [{i + 1}/{len(dataloader)}]', flush=True)

    out = dict(phi=np.stack(all_phi).astype(np.float32),
               pps=np.stack(all_pps).astype(np.float32),
               gts=np.array(all_gts, dtype=np.int64),
               snrs=np.array(all_snrs),
               split=a.split, checkpoint=osp.abspath(a.checkpoint))
    os.makedirs(osp.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'wrote {a.out}: phi {out["phi"].shape}, pps {out["pps"].shape}, '
          f'gts {out["gts"].shape}', flush=True)


if __name__ == '__main__':
    main()
