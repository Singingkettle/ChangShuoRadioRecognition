# Copyright (c) ShuoChang. All rights reserved.
import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from csrr.registry import MODELS
from csrr.structures import DataSample
from .cross_entropy_loss import soft_cross_entropy
from .utils import weight_reduce_loss


def _as_tensor_1d(values, device, dtype=torch.float32):
    tensor = torch.as_tensor(values, device=device, dtype=dtype)
    return tensor.flatten()


_ARRAY_CACHE = {}


def _load_array(source: Union[str, Path]):
    source = Path(source)
    cache_key = str(source.expanduser().resolve())
    if cache_key in _ARRAY_CACHE:
        return _ARRAY_CACHE[cache_key]
    suffix = source.suffix.lower()
    if suffix == '.npy':
        payload = np.load(source)
        _ARRAY_CACHE[cache_key] = payload
        return payload
    if suffix in {'.pkl', '.pickle'}:
        with source.open('rb') as f:
            payload = pickle.load(f)
        _ARRAY_CACHE[cache_key] = payload
        return payload
    if suffix == '.npz':
        with np.load(source, allow_pickle=False) as data:
            payload = {key: data[key] for key in data.files}
        _ARRAY_CACHE[cache_key] = payload
        return payload
    if suffix in {'.pt', '.pth'}:
        payload = torch.load(source, map_location='cpu')
        _ARRAY_CACHE[cache_key] = payload
        return payload
    raise ValueError(f'Unsupported RCPS source file: {source}')


def _get_sample_value(sample: DataSample, key: str):
    value = sample.get(key, None)
    if value is None and hasattr(sample, key):
        value = getattr(sample, key)
    if value is None:
        raise KeyError(
            f'RCPS requires reliability key "{key}" in data_samples. '
            'Add it through PackInputs(meta_keys=(..., "snr")).')
    if isinstance(value, torch.Tensor):
        value = value.detach().flatten()[0]
    return value


def collect_reliability(data_samples: List[DataSample], key: str,
                        device: torch.device) -> torch.Tensor:
    """Collect reliability metadata from a batch of DataSample objects."""
    values = [_get_sample_value(sample, key) for sample in data_samples]
    return _as_tensor_1d(values, device=device)


def collect_sample_indices(data_samples: List[DataSample],
                           key: str,
                           device: torch.device) -> torch.Tensor:
    """Collect sample indices required by sample-posterior RCPS."""
    values = [_get_sample_value(sample, key) for sample in data_samples]
    return torch.as_tensor(values, device=device, dtype=torch.long).flatten()


def map_reliability(raw_reliability: torch.Tensor,
                    cfg: Optional[Dict] = None) -> torch.Tensor:
    """Map a raw reliability coordinate such as SNR to [0, 1]."""
    cfg = cfg or dict(type='identity')
    map_type = cfg.get('type', 'identity')

    if map_type == 'identity':
        reliability = raw_reliability.float()
    elif map_type == 'linear':
        min_value = float(cfg['min'])
        max_value = float(cfg['max'])
        if math.isclose(max_value, min_value):
            raise ValueError('linear reliability_map requires max != min.')
        reliability = (raw_reliability.float() - min_value) / (max_value - min_value)
    else:
        raise ValueError(f'Unsupported reliability map type: {map_type}')

    return reliability.clamp_(0.0, 1.0)


def build_epsilon(reliability: torch.Tensor,
                  cfg: Optional[Dict] = None) -> torch.Tensor:
    """Build the RCPS interpolation strength epsilon(r)."""
    cfg = cfg or dict(type='power', max=1.0, gamma=1.0)
    eps_type = cfg.get('type', 'power')

    if eps_type == 'power':
        eps_max = float(cfg.get('max', 1.0))
        gamma = float(cfg.get('gamma', 1.0))
        epsilon = eps_max * torch.pow(1.0 - reliability, gamma)
    elif eps_type == 'constant':
        epsilon = torch.full_like(reliability, float(cfg.get('value', 0.0)))
    elif eps_type in {'retention_power', 'power_retention'}:
        eps_max = float(cfg.get('max', 1.0))
        gamma = float(cfg.get('gamma', 1.0))
        retain_min = float(cfg.get('retain_min', 0.8))
        transition = float(cfg.get('transition', 0.0))
        epsilon = eps_max * torch.pow(1.0 - reliability, gamma)
        if transition <= 0.0:
            epsilon = torch.where(
                reliability >= retain_min,
                torch.zeros_like(epsilon),
                epsilon)
        else:
            gate = ((retain_min - reliability) / transition).clamp(0.0, 1.0)
            epsilon = epsilon * gate
    elif eps_type in {'low_reliability_power', 'low_power'}:
        eps_max = float(cfg.get('max', 1.0))
        gamma = float(cfg.get('gamma', 1.0))
        cutoff = float(cfg.get('cutoff', 0.25))
        if cutoff <= 0.0:
            raise ValueError('low_reliability_power requires cutoff > 0.')
        scaled = ((cutoff - reliability) / cutoff).clamp(0.0, 1.0)
        epsilon = eps_max * torch.pow(scaled, gamma)
    elif eps_type in {'table', 'piecewise_linear', 'entropy_match'}:
        if 'source' in cfg:
            payload = _load_array(cfg['source'])
            if isinstance(payload, dict):
                bins = payload.get('bins')
                values = payload.get('values', payload.get('epsilon'))
            else:
                raise ValueError('Epsilon table source must contain "bins" and "values".')
        else:
            bins = cfg.get('bins')
            values = cfg.get('values', cfg.get('epsilon'))
        if bins is None or values is None:
            raise ValueError('Epsilon table requires bins and values.')
        bins = torch.as_tensor(bins, dtype=reliability.dtype, device=reliability.device).flatten()
        values = torch.as_tensor(values, dtype=reliability.dtype, device=reliability.device).flatten()
        if bins.numel() != values.numel() or bins.numel() < 2:
            raise ValueError('Epsilon table bins and values must have the same length >= 2.')
        order = torch.argsort(bins)
        bins = bins[order]
        values = values[order]
        idx = torch.bucketize(reliability, bins).clamp(1, bins.numel() - 1)
        left = bins[idx - 1]
        right = bins[idx]
        denom = (right - left).clamp_min(1e-12)
        alpha = ((reliability - left) / denom).clamp(0.0, 1.0)
        epsilon = values[idx - 1] * (1.0 - alpha) + values[idx] * alpha
        epsilon = torch.where(reliability <= bins[0], values[0], epsilon)
        epsilon = torch.where(reliability >= bins[-1], values[-1], epsilon)
    else:
        raise ValueError(f'Unsupported epsilon type: {eps_type}')

    return epsilon.clamp_(0.0, 1.0)


def _normalize_distribution(dist: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    dist = dist.clamp_min(eps)
    return dist / dist.sum(dim=-1, keepdim=True).clamp_min(eps)


def _build_uniform_base(label: torch.Tensor, num_classes: int) -> torch.Tensor:
    return label.new_full((label.size(0), num_classes), 1.0 / num_classes, dtype=torch.float32)


def _build_prior_base(label: torch.Tensor, num_classes: int, cfg: Dict) -> torch.Tensor:
    device = label.device
    if 'prior' in cfg:
        prior = torch.as_tensor(cfg['prior'], dtype=torch.float32, device=device)
    elif 'source' in cfg:
        prior = torch.as_tensor(_load_array(cfg['source']), dtype=torch.float32, device=device)
    else:
        prior = torch.ones(num_classes, dtype=torch.float32, device=device)
    if prior.numel() != num_classes:
        raise ValueError(f'Prior has {prior.numel()} entries, expected {num_classes}.')
    prior = _normalize_distribution(prior.reshape(1, -1))
    return prior.expand(label.size(0), -1)


def _build_confusion_base(label: torch.Tensor, num_classes: int, cfg: Dict,
                          reliability: Optional[torch.Tensor] = None) -> torch.Tensor:
    device = label.device
    if 'matrix' in cfg:
        matrix = torch.as_tensor(cfg['matrix'], dtype=torch.float32, device=device)
    elif 'source' in cfg:
        matrix = torch.as_tensor(_load_array(cfg['source']), dtype=torch.float32, device=device)
    else:
        raise ValueError('Confusion-aware RCPS base requires "matrix" or "source".')

    if matrix.shape != (num_classes, num_classes):
        raise ValueError(f'Confusion matrix shape {tuple(matrix.shape)} does not match ({num_classes}, {num_classes}).')

    laplace = float(cfg.get('laplace', 1e-4))
    temperature = float(cfg.get('temperature', 1.0))
    base = matrix[label.long()].float().clamp_min(0.0) + laplace
    if not math.isclose(temperature, 1.0):
        base = torch.pow(base, 1.0 / temperature)
    base = _normalize_distribution(base)

    prior_blend = float(cfg.get('prior_blend', 0.0))
    if prior_blend > 0.0 and reliability is not None:
        prior_cfg = cfg.get('prior', dict(type='uniform'))
        if isinstance(prior_cfg, (list, tuple)):
            prior_cfg = dict(type='prior', prior=prior_cfg)
        if prior_cfg.get('type', 'uniform') == 'uniform':
            prior = _build_uniform_base(label, num_classes)
        else:
            prior = _build_prior_base(label, num_classes, prior_cfg)
        alpha = (prior_blend * (1.0 - reliability)).clamp(0.0, 1.0).reshape(-1, 1)
        base = (1.0 - alpha) * base + alpha * prior
        base = _normalize_distribution(base)
    return base



def _build_reliability_confusion_base(label: torch.Tensor, num_classes: int,
                                      cfg: Dict,
                                      reliability: Optional[torch.Tensor] = None) -> torch.Tensor:
    if reliability is None:
        raise ValueError('Reliability-conditioned confusion base requires reliability.')
    device = label.device
    if 'source' in cfg:
        payload = _load_array(cfg['source'])
    else:
        payload = cfg

    if isinstance(payload, dict):
        if 'base' in payload:
            matrix = torch.as_tensor(payload['base'], dtype=torch.float32, device=device)
        elif 'matrix' in payload:
            matrix = torch.as_tensor(payload['matrix'], dtype=torch.float32, device=device)
        else:
            raise ValueError('Reliability confusion payload requires "base" or "matrix".')
        if 'bins' in cfg:
            bins = torch.as_tensor(cfg['bins'], dtype=torch.float32, device=device)
        elif 'bins' in payload:
            bins = torch.as_tensor(payload['bins'], dtype=torch.float32, device=device)
        else:
            raise ValueError('Reliability confusion payload requires "bins".')
    else:
        matrix = torch.as_tensor(payload, dtype=torch.float32, device=device)
        if 'bins' not in cfg:
            raise ValueError('Reliability confusion array source requires cfg["bins"].')
        bins = torch.as_tensor(cfg['bins'], dtype=torch.float32, device=device)

    if matrix.ndim != 3 or matrix.shape[1:] != (num_classes, num_classes):
        raise ValueError(
            f'Reliability confusion base shape {tuple(matrix.shape)} must be (B, {num_classes}, {num_classes}).')
    if bins.ndim != 1 or bins.numel() != matrix.shape[0]:
        raise ValueError('Reliability bins must be 1-D and match the first base dimension.')

    distances = torch.abs(reliability.reshape(-1, 1) - bins.reshape(1, -1))
    bin_idx = distances.argmin(dim=1)
    selected = matrix[bin_idx]
    base = selected[torch.arange(label.size(0), device=device), label.long()].float()

    laplace = float(cfg.get('laplace', 0.0))
    if laplace > 0.0:
        base = base + laplace
    temperature = float(cfg.get('temperature', 1.0))
    if not math.isclose(temperature, 1.0):
        base = torch.pow(base.clamp_min(1e-12), 1.0 / temperature)
    base = _normalize_distribution(base)

    prior_blend = float(cfg.get('prior_blend', 0.0))
    if prior_blend > 0.0:
        prior_cfg = cfg.get('prior', dict(type='uniform'))
        if isinstance(prior_cfg, (list, tuple)):
            prior_cfg = dict(type='prior', prior=prior_cfg)
        if prior_cfg.get('type', 'uniform') == 'uniform':
            prior = _build_uniform_base(label, num_classes)
        else:
            prior = _build_prior_base(label, num_classes, prior_cfg)
        alpha = (prior_blend * (1.0 - reliability)).clamp(0.0, 1.0).reshape(-1, 1)
        base = _normalize_distribution((1.0 - alpha) * base + alpha * prior)
    return base


def _build_sample_posterior_base(label: torch.Tensor,
                                 num_classes: int,
                                 cfg: Dict,
                                 reliability: Optional[torch.Tensor] = None,
                                 sample_indices: Optional[torch.Tensor] = None,
                                 posterior_indices: Optional[torch.Tensor] = None,
                                 posterior_probs: Optional[torch.Tensor] = None,
                                 posterior_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Lookup sample-adaptive posterior bases by sample index.

    The source artifact must be generated only from train/validation evidence.
    Test-set sample posteriors are intentionally unsupported for training.
    """
    if sample_indices is None:
        raise ValueError('Sample-posterior RCPS requires packed sample_idx metadata.')
    device = label.device

    if posterior_indices is None or posterior_probs is None:
        if 'source' not in cfg:
            raise ValueError('Sample-posterior RCPS base requires "source".')
        payload = _load_array(cfg['source'])
        if not isinstance(payload, dict):
            raise ValueError('Sample-posterior source must be an npz/dict payload.')
        if 'sample_idx' not in payload or 'probs' not in payload:
            raise ValueError('Sample-posterior source requires "sample_idx" and "probs".')
        posterior_indices = torch.as_tensor(payload['sample_idx'], dtype=torch.long, device=device).flatten()
        posterior_probs = torch.as_tensor(payload['probs'], dtype=torch.float32, device=device)
        posterior_labels = (
            torch.as_tensor(payload['label'], dtype=torch.long, device=device).flatten()
            if 'label' in payload else None)
        order = torch.argsort(posterior_indices)
        posterior_indices = posterior_indices[order]
        posterior_probs = posterior_probs[order]
        if posterior_labels is not None:
            posterior_labels = posterior_labels[order]

    if posterior_probs.ndim != 2 or posterior_probs.size(1) != num_classes:
        raise ValueError(
            f'Sample-posterior probs shape {tuple(posterior_probs.shape)} '
            f'does not match num_classes={num_classes}.')

    sample_indices = sample_indices.long().flatten()
    positions = torch.searchsorted(posterior_indices, sample_indices)
    valid = positions < posterior_indices.numel()
    safe_positions = positions.clamp(max=max(posterior_indices.numel() - 1, 0))
    matched = valid & (posterior_indices[safe_positions] == sample_indices)
    if not torch.all(matched):
        missing = sample_indices[~matched][:8].detach().cpu().tolist()
        raise KeyError(f'Sample-posterior source is missing sample_idx values: {missing}')

    if posterior_labels is not None:
        expected = posterior_labels[safe_positions]
        if not torch.all(expected == label.long().flatten()):
            bad = sample_indices[expected != label.long().flatten()][:8].detach().cpu().tolist()
            raise ValueError(f'Sample-posterior labels do not match batch labels for sample_idx: {bad}')

    base = posterior_probs[safe_positions].float()
    laplace = float(cfg.get('laplace', 0.0))
    if laplace > 0.0:
        base = base + laplace
    temperature = float(cfg.get('temperature', 1.0))
    if not math.isclose(temperature, 1.0):
        base = torch.pow(base.clamp_min(1e-12), 1.0 / temperature)
    base = _normalize_distribution(base)

    prior_blend = float(cfg.get('prior_blend', 0.0))
    if prior_blend > 0.0 and reliability is not None:
        prior_cfg = cfg.get('prior', dict(type='uniform'))
        if isinstance(prior_cfg, (list, tuple)):
            prior_cfg = dict(type='prior', prior=prior_cfg)
        if prior_cfg.get('type', 'uniform') == 'uniform':
            prior = _build_uniform_base(label, num_classes)
        else:
            prior = _build_prior_base(label, num_classes, prior_cfg)
        alpha = (prior_blend * (1.0 - reliability)).clamp(0.0, 1.0).reshape(-1, 1)
        base = _normalize_distribution((1.0 - alpha) * base + alpha * prior)
    return base

def build_base_distribution(label: torch.Tensor,
                            num_classes: int,
                            cfg: Optional[Dict] = None,
                            reliability: Optional[torch.Tensor] = None,
                            sample_indices: Optional[torch.Tensor] = None,
                            posterior_indices: Optional[torch.Tensor] = None,
                            posterior_probs: Optional[torch.Tensor] = None,
                            posterior_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
    cfg = cfg or dict(type='uniform')
    base_type = cfg.get('type', 'uniform')
    if base_type == 'uniform':
        return _build_uniform_base(label, num_classes)
    if base_type == 'prior':
        return _build_prior_base(label, num_classes, cfg)
    if base_type == 'confusion':
        return _build_confusion_base(label, num_classes, cfg, reliability)
    if base_type in {'reliability_confusion', 'bin_confusion', 'posterior_table'}:
        return _build_reliability_confusion_base(label, num_classes, cfg, reliability)
    if base_type in {'sample_posterior', 'dpc_sample_posterior'}:
        return _build_sample_posterior_base(
            label,
            num_classes,
            cfg,
            reliability=reliability,
            sample_indices=sample_indices,
            posterior_indices=posterior_indices,
            posterior_probs=posterior_probs,
            posterior_labels=posterior_labels)
    raise ValueError(f'Unsupported RCPS base type: {base_type}')


def build_sample_weight(reliability: torch.Tensor,
                        cfg: Optional[Dict] = None) -> Optional[torch.Tensor]:
    cfg = cfg or dict(type='none')
    weight_type = cfg.get('type', 'none')
    if weight_type == 'none':
        return None
    if weight_type == 'reliability_power':
        gamma = float(cfg.get('gamma', 1.0))
        min_weight = float(cfg.get('min', 0.0))
        return (min_weight + (1.0 - min_weight) * torch.pow(reliability, gamma)).clamp_min(0.0)
    raise ValueError(f'Unsupported sample weight type: {weight_type}')


def build_rcps_targets(label: torch.Tensor,
                       raw_reliability: torch.Tensor,
                       num_classes: int,
                       reliability_map: Optional[Dict] = None,
                       epsilon: Optional[Dict] = None,
                       base: Optional[Dict] = None,
                       sample_indices: Optional[torch.Tensor] = None,
                       posterior_indices: Optional[torch.Tensor] = None,
                       posterior_probs: Optional[torch.Tensor] = None,
                       posterior_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Build reliability-conditioned posterior supervision targets."""
    if label.ndim != 1:
        label = label.flatten()
    if label.numel() != raw_reliability.numel():
        raise ValueError('label and raw_reliability must have the same batch size.')

    reliability = map_reliability(raw_reliability, reliability_map)
    eps = build_epsilon(reliability, epsilon).reshape(-1, 1)
    one_hot = F.one_hot(label.long(), num_classes=num_classes).float()
    base_dist = build_base_distribution(
        label,
        num_classes,
        base,
        reliability,
        sample_indices=sample_indices,
        posterior_indices=posterior_indices,
        posterior_probs=posterior_probs,
        posterior_labels=posterior_labels)
    target = (1.0 - eps) * one_hot + eps * base_dist
    return _normalize_distribution(target)


@MODELS.register_module()
class RCPSCrossEntropyLoss(nn.Module):
    """Reliability-Conditioned Posterior Supervision loss."""

    requires_data_samples = True

    def __init__(self,
                 reliability_key: str = 'snr',
                 reliability_map: Optional[Dict] = None,
                 epsilon: Optional[Dict] = None,
                 base: Optional[Dict] = None,
                 sample_weight: Optional[Dict] = None,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 class_weight: Optional[Sequence[float]] = None):
        super().__init__()
        self.reliability_key = reliability_key
        self.reliability_map = reliability_map or dict(type='identity')
        self.epsilon = epsilon or dict(type='power', max=1.0, gamma=1.0)
        self.base = base or dict(type='uniform')
        self.sample_weight = sample_weight or dict(type='none')
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.sample_index_key = self.base.get('sample_index_key', 'sample_idx')

        if self.base.get('type') in {'sample_posterior', 'dpc_sample_posterior'}:
            if 'source' not in self.base:
                raise ValueError('Sample-posterior RCPS base requires "source".')
            payload = _load_array(self.base['source'])
            if not isinstance(payload, dict):
                raise ValueError('Sample-posterior source must be an npz/dict payload.')
            for required in ('sample_idx', 'probs'):
                if required not in payload:
                    raise ValueError(f'Sample-posterior source missing "{required}".')
            indices = torch.as_tensor(payload['sample_idx'], dtype=torch.long).flatten()
            probs = torch.as_tensor(payload['probs'], dtype=torch.float32)
            if probs.ndim != 2 or probs.size(0) != indices.numel():
                raise ValueError('Sample-posterior probs must have shape (N, C) matching sample_idx.')
            order = torch.argsort(indices)
            self.register_buffer('sample_posterior_indices', indices[order], persistent=False)
            self.register_buffer('sample_posterior_probs', probs[order], persistent=False)
            if 'label' in payload:
                labels = torch.as_tensor(payload['label'], dtype=torch.long).flatten()
                if labels.numel() != indices.numel():
                    raise ValueError('Sample-posterior label must match sample_idx length.')
                self.register_buffer('sample_posterior_labels', labels[order], persistent=False)
            else:
                self.sample_posterior_labels = None
        else:
            self.sample_posterior_indices = None
            self.sample_posterior_probs = None
            self.sample_posterior_labels = None

    def forward(self,
                cls_score: torch.Tensor,
                label: torch.Tensor,
                weight: Optional[torch.Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                data_samples: Optional[List[DataSample]] = None,
                **kwargs) -> torch.Tensor:
        if data_samples is None:
            raise ValueError('RCPSCrossEntropyLoss requires data_samples.')
        if label.ndim != 1:
            raise ValueError('RCPSCrossEntropyLoss expects hard class labels, not gt_score.')

        raw_reliability = collect_reliability(data_samples, self.reliability_key, cls_score.device)
        sample_indices = None
        if self.base.get('type') in {'sample_posterior', 'dpc_sample_posterior'}:
            sample_indices = collect_sample_indices(data_samples, self.sample_index_key, cls_score.device)
        targets = build_rcps_targets(
            label=label,
            raw_reliability=raw_reliability,
            num_classes=cls_score.size(1),
            reliability_map=self.reliability_map,
            epsilon=self.epsilon,
            base=self.base,
            sample_indices=sample_indices,
            posterior_indices=self.sample_posterior_indices,
            posterior_probs=self.sample_posterior_probs,
            posterior_labels=self.sample_posterior_labels)

        reliability = map_reliability(raw_reliability, self.reliability_map)
        rcps_weight = build_sample_weight(reliability, self.sample_weight)
        if weight is not None and rcps_weight is not None:
            weight = weight.to(cls_score.device).float() * rcps_weight
        elif rcps_weight is not None:
            weight = rcps_weight

        class_weight = None
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)

        reduction = reduction_override if reduction_override else self.reduction
        return self.loss_weight * soft_cross_entropy(
            cls_score,
            targets,
            weight=weight,
            reduction=reduction,
            class_weight=class_weight,
            avg_factor=avg_factor)


@MODELS.register_module()
class LabelSmoothingCrossEntropyLoss(nn.Module):
    """Static label smoothing baseline implemented as soft cross entropy."""

    def __init__(self,
                 smoothing: float = 0.1,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 class_weight: Optional[Sequence[float]] = None):
        super().__init__()
        self.smoothing = float(smoothing)
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

    def forward(self,
                cls_score: torch.Tensor,
                label: torch.Tensor,
                weight: Optional[torch.Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> torch.Tensor:
        num_classes = cls_score.size(1)
        one_hot = F.one_hot(label.long().flatten(), num_classes=num_classes).float()
        uniform = one_hot.new_full(one_hot.shape, 1.0 / num_classes)
        targets = (1.0 - self.smoothing) * one_hot + self.smoothing * uniform
        class_weight = None
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        reduction = reduction_override if reduction_override else self.reduction
        return self.loss_weight * soft_cross_entropy(
            cls_score,
            targets,
            weight=weight,
            reduction=reduction,
            class_weight=class_weight,
            avg_factor=avg_factor)


@MODELS.register_module()
class ConfidencePenaltyLoss(nn.Module):
    """Cross entropy with a confidence penalty on low-entropy predictions."""

    def __init__(self,
                 beta: float = 0.1,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0):
        super().__init__()
        self.beta = float(beta)
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                cls_score: torch.Tensor,
                label: torch.Tensor,
                weight: Optional[torch.Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> torch.Tensor:
        ce = F.cross_entropy(cls_score, label.long().flatten(), reduction='none')
        log_prob = F.log_softmax(cls_score, dim=-1)
        prob = log_prob.exp()
        penalty = (prob * log_prob).sum(dim=-1)
        loss = ce + self.beta * penalty
        reduction = reduction_override if reduction_override else self.reduction
        loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
        return self.loss_weight * loss
