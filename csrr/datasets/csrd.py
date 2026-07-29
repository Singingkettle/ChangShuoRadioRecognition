# Copyright (c) Shuo Chang. All Rights Reserved.
"""Datasets for the CSRD (ChangShuoRadioData / ``twc`` profile, a.k.a. CRML23)
multi-signal frames used by the JDM method.

On-disk layout (one directory per channel/impairment configuration)::

    <data_root>/
        v1/
            anno/000001.json ... 001000.json
            sequence_data/iq/000001.mat ... 001000.mat
        v2/ ...

Each ``.mat`` stores ``signal_data`` of shape ``(num_signals, 2, L)`` — the
per-signal passband I/Q components whose sum is the received frame. Each
annotation JSON stores parallel per-signal arrays (``center_frequency``,
``bandwidth``, ``modulation``, ``snr``, ``channel``, ...).

No split files exist on disk, so both datasets perform a deterministic seeded
split of every version's entries (default 50/10/40 train/validation/test —
the repo-wide convention, cf. ``tools/convert_datasets``).
"""
import os
import os.path as osp
import random
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
from mmengine.fileio import load

from csrr.registry import DATASETS
from .base_dataset import BaseClassificationDataset

DEFAULT_SPLIT_RATIOS = (0.5, 0.1, 0.4)
SPLIT_NAMES = ('train', 'validation', 'test')


def _list_versions(data_root: str,
                   versions: Union[str, Sequence[str], None]) -> List[str]:
    if versions is None or versions == 'all':
        versions = sorted(
            (d for d in os.listdir(data_root)
             if osp.isdir(osp.join(data_root, d)) and d.startswith('v')),
            key=lambda d: int(d[1:]))
    elif isinstance(versions, str):
        versions = [versions]
    return list(versions)


def _split_indices(num_items: int, split: str, split_ratios, seed: int
                   ) -> List[int]:
    assert split in SPLIT_NAMES, \
        f'split must be one of {SPLIT_NAMES}, got {split!r}'
    indices = list(range(num_items))
    random.Random(seed).shuffle(indices)
    num_train = int(split_ratios[0] * num_items)
    num_val = int(split_ratios[1] * num_items)
    if split == 'train':
        return indices[:num_train]
    if split == 'validation':
        return indices[num_train:num_train + num_val]
    return indices[num_train + num_val:]


class _CSRDBase(BaseClassificationDataset):
    """Shared CSRD scanning/splitting logic."""

    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 versions: Union[str, Sequence[str], None] = None,
                 split_ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
                 split_seed: int = 0,
                 frame_length: int = 1200,
                 metainfo: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000) -> None:
        self.split = split
        self.versions = versions
        self.split_ratios = tuple(split_ratios)
        self.split_seed = split_seed
        self.frame_length = frame_length

        super().__init__(
            ann_file='',
            metainfo=metainfo,
            data_root=data_root,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch)

    def _scan_entries(self) -> List[dict]:
        """Collect the annotation dicts of this split over all versions."""
        entries = []
        classes = set()
        for version in _list_versions(self.data_root, self.versions):
            anno_dir = osp.join(self.data_root, version, 'anno')
            anno_files = sorted(os.listdir(anno_dir))
            for idx in _split_indices(
                    len(anno_files), self.split, self.split_ratios,
                    self.split_seed):
                anno = load(osp.join(anno_dir, anno_files[idx]))
                anno['version'] = version
                anno['iq_path'] = osp.join(
                    self.data_root, version, 'sequence_data', 'iq',
                    anno['file_name'])
                entries.append(anno)
                classes.update(anno['modulation'])
        if 'classes' not in self._metainfo:
            self._metainfo['classes'] = tuple(sorted(classes))
        return entries


@DATASETS.register_module()
class CSRDDetectionDataset(_CSRDBase):
    """Frame-level dataset for the JDM detection module.

    One item per frame. Annotations are converted to 1-D frequency intervals
    in FFT-bin units: ``bin = (f / sample_rate + 0.5) * frame_length``.

    Args:
        data_root (str): directory holding the ``v*`` version folders.
        split (str): 'train' / 'validation' / 'test'.
        versions (str | Sequence[str] | None): version folders to use
            (None or 'all' = every ``v*`` directory).
        split_ratios (Sequence[float]): train/val/test fractions applied
            per version with a fixed seed.
        frame_length (int): FFT length L of the stored frames.
    """

    def load_data_list(self) -> List[dict]:
        class_to_idx = None
        data_list = []
        for anno in self._scan_entries():
            if class_to_idx is None:
                class_to_idx = {
                    name: i for i, name in enumerate(self.CLASSES)}
            sample_rate = anno['sample_rate'][0] if anno['sample_rate'] \
                else 150000
            boxes = []
            for cf, bw in zip(anno['center_frequency'], anno['bandwidth']):
                left = (cf - bw / 2) / sample_rate + 0.5
                right = (cf + bw / 2) / sample_rate + 0.5
                boxes.append([left * self.frame_length,
                              right * self.frame_length])
            boxes = np.array(boxes, dtype=np.float32).reshape(-1, 2)
            labels = np.array(
                [class_to_idx[m] for m in anno['modulation']],
                dtype=np.int64)
            data_list.append(
                dict(
                    iq_path=anno['iq_path'],
                    file_name=anno['file_name'],
                    version=anno['version'],
                    sample_rate=sample_rate,
                    frame_length=self.frame_length,
                    gt_boxes=boxes,
                    gt_box_labels=labels,
                    snr=anno['snr'],
                    channel=anno['channel']))
        return data_list


@DATASETS.register_module()
class CSRDModulationDetPropDataset(_CSRDBase):
    """Signal-level AMC dataset with detector proposals and hard negatives.

    Extends :class:`CSRDModulationDataset` with proposal-cache lookups and
    optional unmatched detector proposals per frame for hard-negative mining.

    Args:
        proposal_cache (str): JSON cache from ``precompute_amc_proposals.py``.
        include_hard_negatives (bool): append unmatched proposal crops on the
            train split. Defaults to False.
        max_hard_neg_per_frame (int): cap hard negatives added per frame.
    """

    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 versions: Union[str, Sequence[str], None] = None,
                 split_ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
                 split_seed: int = 0,
                 frame_length: int = 1200,
                 metainfo: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 proposal_cache: str = '',
                 include_hard_negatives: bool = False,
                 max_hard_neg_per_frame: int = 3) -> None:
        import json
        with open(proposal_cache, encoding='utf-8') as f:
            self._proposal_cache = json.load(f)
        self.include_hard_negatives = include_hard_negatives
        self.max_hard_neg_per_frame = max_hard_neg_per_frame
        super().__init__(
            data_root=data_root,
            split=split,
            versions=versions,
            split_ratios=split_ratios,
            split_seed=split_seed,
            frame_length=frame_length,
            metainfo=metainfo,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch)

    def load_data_list(self) -> List[dict]:
        class_to_idx = None
        data_list = []
        for anno in self._scan_entries():
            if class_to_idx is None:
                class_to_idx = {
                    name: i for i, name in enumerate(self.CLASSES)}
            sample_rate = anno['sample_rate'][0] if anno['sample_rate'] \
                else 150000
            file_name = anno['file_name']
            for sig_idx, modulation in enumerate(anno['modulation']):
                data_list.append(
                    dict(
                        iq_path=anno['iq_path'],
                        file_name=file_name,
                        version=anno['version'],
                        sample_rate=sample_rate,
                        frame_length=self.frame_length,
                        signal_index=sig_idx,
                        is_hard_negative=False,
                        center_frequency=anno['center_frequency'][sig_idx],
                        bandwidth=anno['bandwidth'][sig_idx],
                        snr=anno['snr'][sig_idx],
                        channel=anno['channel'][sig_idx],
                        gt_label=np.array(
                            class_to_idx[modulation], dtype=np.int64)))
            if self.include_hard_negatives:
                unmatched = self._proposal_cache.get(
                    file_name, {}).get('_unmatched', [])
                for neg_idx in range(
                        min(len(unmatched), self.max_hard_neg_per_frame)):
                    data_list.append(
                        dict(
                            iq_path=anno['iq_path'],
                            file_name=file_name,
                            version=anno['version'],
                            sample_rate=sample_rate,
                            frame_length=self.frame_length,
                            is_hard_negative=True,
                            hard_neg_index=neg_idx))
        return data_list


@DATASETS.register_module()
class CSRDModulationDataset(_CSRDBase):
    """Signal-level dataset for the JDM classification module.

    One item per annotated signal inside a frame; the pipeline extracts the
    single-signal baseband crop (:class:`CSRDSignalToBaseband`) that the
    paper's classification module consumes.
    """

    def load_data_list(self) -> List[dict]:
        class_to_idx = None
        data_list = []
        for anno in self._scan_entries():
            if class_to_idx is None:
                class_to_idx = {
                    name: i for i, name in enumerate(self.CLASSES)}
            sample_rate = anno['sample_rate'][0] if anno['sample_rate'] \
                else 150000
            for sig_idx, modulation in enumerate(anno['modulation']):
                data_list.append(
                    dict(
                        iq_path=anno['iq_path'],
                        file_name=anno['file_name'],
                        version=anno['version'],
                        sample_rate=sample_rate,
                        frame_length=self.frame_length,
                        signal_index=sig_idx,
                        center_frequency=anno['center_frequency'][sig_idx],
                        bandwidth=anno['bandwidth'][sig_idx],
                        snr=anno['snr'][sig_idx],
                        channel=anno['channel'][sig_idx],
                        gt_label=np.array(
                            class_to_idx[modulation], dtype=np.int64)))
        return data_list
