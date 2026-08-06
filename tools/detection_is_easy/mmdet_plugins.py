# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.transforms import BaseTransform
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.registry import MODELS, TRANSFORMS

from iqdet_complex import ComplexGaborFilterbank, ComplexTemporalConvFilterbank, MultiScaleComplexGaborFilterbank
from iqdet_complex import NativeComplexConv1d, ComplexNormAct


_MEMMAP_CACHE: dict[tuple[str, str], np.memmap] = {}


def _sample_index_from_path(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem.rsplit("_", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"Cannot parse sample index from {path}") from exc


def _split_from_img_path(path: Path) -> str:
    # Expected MMDet path: <data_root>/<split>/tensors/<sample>.npy
    if path.parent.name == "tensors":
        return path.parent.parent.name
    return path.parent.name


def _open_memmap(path: Path, *, mmap_mode: str = "r") -> np.memmap:
    key = (str(path), mmap_mode)
    if key not in _MEMMAP_CACHE:
        _MEMMAP_CACHE[key] = np.load(path, mmap_mode=mmap_mode)
    return _MEMMAP_CACHE[key]


try:
    from mmcv.cnn import ConvModule
    from mmdet.models.backbones.csp_darknet import CSPDarknet, Focus
    from mmdet.models.backbones.cspnext import CSPNeXt
except ModuleNotFoundError as exc:
    _BACKBONE_IMPORT_ERROR = exc

    class _MissingBackbone(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "Complex STFT custom backbones require full MMDetection/MMCV backbone imports."
            ) from _BACKBONE_IMPORT_ERROR

    def _missing_layer(*args, **kwargs):
        raise ModuleNotFoundError(
            "Complex STFT custom backbones require full MMDetection/MMCV backbone imports."
        ) from _BACKBONE_IMPORT_ERROR

    ConvModule = _missing_layer
    CSPDarknet = _MissingBackbone
    Focus = _missing_layer
    CSPNeXt = _MissingBackbone
else:
    _BACKBONE_IMPORT_ERROR = None


def _samplelist_boxtype2tensor(data_samples) -> None:
    from mmdet.models.utils.misc import samplelist_boxtype2tensor

    samplelist_boxtype2tensor(data_samples)


@HOOKS.register_module()
class FreezeDetectorExceptFilterbankHook(Hook):
    """Freeze detector parameters while keeping the raw-IQ filterbank trainable."""

    priority = "VERY_HIGH"

    def __init__(self, trainable_prefix: str = "data_preprocessor.filterbank") -> None:
        self.trainable_prefix = trainable_prefix

    def _apply_freeze(self, runner) -> None:
        model = runner.model
        for name, param in model.named_parameters():
            param.requires_grad_(name.startswith(self.trainable_prefix))
        for name, module in model.named_modules():
            if name and not name.startswith(self.trainable_prefix):
                module.eval()

    def before_train(self, runner) -> None:
        self._apply_freeze(runner)

    def before_train_epoch(self, runner) -> None:
        self._apply_freeze(runner)


@HOOKS.register_module()
class FreezeExceptPrefixesHook(Hook):
    """Freeze all model parameters except modules with selected prefixes."""

    priority = "VERY_HIGH"

    def __init__(self, trainable_prefixes: str | list[str] | tuple[str, ...]) -> None:
        if isinstance(trainable_prefixes, str):
            trainable_prefixes = [trainable_prefixes]
        self.trainable_prefixes = tuple(str(prefix) for prefix in trainable_prefixes)
        if not self.trainable_prefixes:
            raise ValueError("FreezeExceptPrefixesHook requires at least one trainable prefix")

    def _is_trainable(self, name: str) -> bool:
        return any(name == prefix or name.startswith(prefix + ".") for prefix in self.trainable_prefixes)

    def _apply_freeze(self, runner) -> None:
        model = runner.model
        for name, param in model.named_parameters():
            param.requires_grad_(self._is_trainable(name))
        for name, module in model.named_modules():
            if name and not self._is_trainable(name):
                module.eval()

    def before_train(self, runner) -> None:
        self._apply_freeze(runner)

    def before_train_epoch(self, runner) -> None:
        self._apply_freeze(runner)

    def before_train_iter(self, runner, batch_idx: int, data_batch=None) -> None:
        self._apply_freeze(runner)


@TRANSFORMS.register_module()
class LoadComplexStftFromFile(BaseTransform):
    """Load a complex STFT tensor saved as real/imag channels.

    The tensor is stored as ``[2, frequency, time]`` and converted to the
    MMDetection image convention ``[frequency, time, 2]`` without quantizing or
    rendering the spectrum as an image.
    """

    def __init__(self, to_float32: bool = True, expected_channels: int | None = None) -> None:
        self.to_float32 = to_float32
        self.expected_channels = expected_channels

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        filename = Path(results["img_path"])
        tensor = np.load(filename)
        if isinstance(tensor, np.lib.npyio.NpzFile):
            tensor = tensor["stft"]
        tensor = np.asarray(tensor)
        if tensor.ndim != 3:
            raise ValueError(f"Expected [C, F, T] STFT tensor, got {tensor.shape} from {filename}")
        if self.expected_channels is not None and tensor.shape[0] != self.expected_channels:
            raise ValueError(
                f"Expected [{self.expected_channels}, F, T] STFT tensor, got {tensor.shape} from {filename}"
            )
        if not np.all(np.isfinite(tensor)):
            raise ValueError(f"Complex STFT tensor contains NaN/Inf: {filename}")

        image = np.transpose(tensor, (1, 2, 0))
        if self.to_float32:
            image = image.astype(np.float32, copy=False)

        results["img"] = image
        results["img_shape"] = image.shape[:2]
        results["ori_shape"] = image.shape[:2]
        results["scale_factor"] = (1.0, 1.0)
        results["complex_stft_shape"] = tensor.shape
        return results


@TRANSFORMS.register_module()
class LoadComplexStftWithLogMagnitudeFromFile(BaseTransform):
    """Load [real, imag] STFT and append log magnitude as a third channel.

    This is mathematically equivalent to precomputing STFT3 tensors with
    ``[real, imag, log1p(abs(real + j imag))]`` but avoids writing a second
    copy of every large STFT tensor to disk.
    """

    def __init__(self, to_float32: bool = True, expected_input_channels: int = 2) -> None:
        self.to_float32 = to_float32
        self.expected_input_channels = int(expected_input_channels)

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        filename = Path(results["img_path"])
        tensor = np.load(filename)
        if isinstance(tensor, np.lib.npyio.NpzFile):
            tensor = tensor["stft"]
        tensor = np.asarray(tensor)
        if tensor.ndim != 3:
            raise ValueError(f"Expected [C, F, T] STFT tensor, got {tensor.shape} from {filename}")
        if tensor.shape[0] != self.expected_input_channels:
            raise ValueError(
                f"Expected [{self.expected_input_channels}, F, T] STFT tensor, got {tensor.shape} from {filename}"
            )
        if tensor.shape[0] != 2:
            raise ValueError("Dynamic log-magnitude STFT loader currently requires real/imag input channels")
        if not np.all(np.isfinite(tensor)):
            raise ValueError(f"Complex STFT tensor contains NaN/Inf: {filename}")

        real = tensor[0].astype(np.float32, copy=False)
        imag = tensor[1].astype(np.float32, copy=False)
        logmag = np.log1p(np.sqrt(np.square(real) + np.square(imag))).astype(np.float32, copy=False)
        out = np.stack([real, imag, logmag], axis=0)
        image = np.transpose(out, (1, 2, 0))
        if self.to_float32:
            image = image.astype(np.float32, copy=False)

        results["img"] = image
        results["img_shape"] = image.shape[:2]
        results["ori_shape"] = image.shape[:2]
        results["scale_factor"] = (1.0, 1.0)
        results["complex_stft_shape"] = out.shape
        return results


@TRANSFORMS.register_module()
class LoadComplexStftMemmapWithLogMagnitude(BaseTransform):
    """Load [real, imag] STFT from split memmaps and append log magnitude."""

    def __init__(self, memmap_root: str, to_float32: bool = True) -> None:
        self.memmap_root = Path(memmap_root)
        self.to_float32 = to_float32

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        filename = Path(results["img_path"])
        split = _split_from_img_path(filename)
        index = _sample_index_from_path(filename)
        mmap = _open_memmap(self.memmap_root / f"{split}.npy")
        tensor = np.asarray(mmap[index])
        if tensor.ndim != 3 or tensor.shape[0] != 2:
            raise ValueError(f"Expected memmap item [2,F,T], got {tensor.shape} from {filename}")
        real = tensor[0].astype(np.float32, copy=False)
        imag = tensor[1].astype(np.float32, copy=False)
        logmag = np.log1p(np.sqrt(np.square(real) + np.square(imag))).astype(np.float32, copy=False)
        out = np.stack([real, imag, logmag], axis=0)
        image = np.transpose(out, (1, 2, 0))
        if self.to_float32:
            image = image.astype(np.float32, copy=False)
        results["img"] = image
        results["img_shape"] = image.shape[:2]
        results["ori_shape"] = image.shape[:2]
        results["scale_factor"] = (1.0, 1.0)
        results["complex_stft_shape"] = out.shape
        results["memmap_index"] = int(index)
        return results


@TRANSFORMS.register_module()
class LoadTensorMemmapFromCOCOStem(BaseTransform):
    """Load a precomputed COCO tensor from a split-level memmap.

    This generic loader is intended for already materialized detector tensors
    such as STFT3 ``[real, imag, log_magnitude]``. It keeps the formal baseline
    as float tensors and avoids per-sample ``.npy`` file opens during training.
    The memmap item is stored as ``[C, H, W]`` and converted to MMDetection's
    image convention ``[H, W, C]``.
    """

    def __init__(
        self,
        memmap_root: str,
        *,
        expected_channels: int | None = None,
        to_float32: bool = True,
    ) -> None:
        self.memmap_root = Path(memmap_root)
        self.expected_channels = int(expected_channels) if expected_channels is not None else None
        self.to_float32 = to_float32

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        filename = Path(results["img_path"])
        split = _split_from_img_path(filename)
        index = _sample_index_from_path(filename)
        mmap = _open_memmap(self.memmap_root / f"{split}.npy")
        tensor = np.asarray(mmap[index])
        if tensor.ndim != 3:
            raise ValueError(f"Expected memmap item [C,H,W], got {tensor.shape} from {filename}")
        if self.expected_channels is not None and tensor.shape[0] != self.expected_channels:
            raise ValueError(
                f"Expected memmap item [{self.expected_channels},H,W], got {tensor.shape} from {filename}"
            )
        if not np.all(np.isfinite(tensor)):
            raise ValueError(f"Tensor memmap item contains NaN/Inf: {filename}")

        image = np.transpose(tensor, (1, 2, 0))
        if self.to_float32:
            image = image.astype(np.float32, copy=False)
        results["img"] = image
        results["img_shape"] = image.shape[:2]
        results["ori_shape"] = image.shape[:2]
        results["scale_factor"] = (1.0, 1.0)
        results["tensor_memmap_shape"] = tensor.shape
        results["complex_stft_shape"] = tensor.shape
        results["memmap_index"] = int(index)
        return results


@TRANSFORMS.register_module()
class LoadRawIQFromCOCOStem(BaseTransform):
    """Load raw IQ while reusing COCO boxes from a tensor export.

    The COCO file name supplies the sample id and split, for example
    ``train/tensors/train_000000.npy``. The raw IQ is then read from
    ``raw_root/train/train_000000.npz``. This lets us train the same detector
    against the same annotations while moving the time-frequency front end into
    the model graph.
    """

    def __init__(
        self,
        raw_root: str,
        *,
        raw_cache_root: str | None = None,
        iq_key: str = "iq",
        target_shape: tuple[int, int] = (512, 512),
        to_float32: bool = True,
    ) -> None:
        self.raw_root = Path(raw_root)
        self.raw_cache_root = Path(raw_cache_root) if raw_cache_root else None
        self.iq_key = iq_key
        self.target_shape = tuple(int(v) for v in target_shape)
        self.to_float32 = to_float32

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        stem_path = Path(results["img_path"])
        sample_id = stem_path.stem
        raw_path = self._resolve_raw_path(stem_path, sample_id)
        raw = self._load_raw_iq(raw_path)
        if np.iscomplexobj(raw):
            iq = np.asarray(raw, dtype=np.complex64)
            pair = np.stack([iq.real, iq.imag], axis=-1)
        else:
            pair = np.asarray(raw, dtype=np.float32)
            if pair.ndim != 2 or pair.shape[-1] != 2:
                raise ValueError(f"Expected raw IQ pair [N, 2], got {pair.shape} from {raw_path}")
        if self.to_float32:
            pair = pair.astype(np.float32, copy=False)
        # PackDetInputs expects HWC. Use W=1 and C=2 so the packed tensor is
        # [2, N, 1], which the raw-IQ preprocessor converts into [B, 2, N].
        image = pair[:, None, :]
        results["img"] = image
        results["img_shape"] = self.target_shape
        results["ori_shape"] = self.target_shape
        results["scale_factor"] = (1.0, 1.0)
        results["raw_iq_path"] = str(raw_path)
        results["raw_iq_shape"] = (2, int(pair.shape[0]))
        return results

    def _load_raw_iq(self, raw_path: Path) -> np.ndarray:
        if raw_path.suffix == ".npy":
            return np.load(raw_path, mmap_mode="r")
        with np.load(raw_path) as npz:
            return np.asarray(npz[self.iq_key], dtype=np.complex64)

    def _resolve_raw_path(self, stem_path: Path, sample_id: str) -> Path:
        # Legacy exports used split/tensors/sample.npy. Latest TorchSig COCO
        # exports use flat sample names such as
        # wideband_clean_train_all_00000000.npy. Support both so raw-IQ and
        # STFT baselines can share the same annotations.
        candidates: list[Path] = []
        roots = [root for root in (self.raw_cache_root, self.raw_root) if root is not None]
        for root in roots:
            if stem_path.parent.parent.name:
                candidates.append(root / stem_path.parent.parent.name / f"{sample_id}.npy")
            candidates.append(root / f"{sample_id}.npy")
            for split_dir in sorted(path for path in root.iterdir() if path.is_dir()) if root.exists() else []:
                candidates.append(split_dir / f"{sample_id}.npy")
                prefix = f"{split_dir.name}_"
                if sample_id.startswith(prefix):
                    candidates.append(split_dir / f"{sample_id}.npy")

        if stem_path.parent.parent.name:
            candidates.append(self.raw_root / stem_path.parent.parent.name / f"{sample_id}.npy")
            candidates.append(self.raw_root / stem_path.parent.parent.name / f"{sample_id}.npz")
        candidates.append(self.raw_root / f"{sample_id}.npy")
        candidates.append(self.raw_root / f"{sample_id}.npz")
        for split_dir in sorted(path for path in self.raw_root.iterdir() if path.is_dir()) if self.raw_root.exists() else []:
            candidates.append(split_dir / f"{sample_id}.npy")
            candidates.append(split_dir / f"{sample_id}.npz")
            prefix = f"{split_dir.name}_"
            if sample_id.startswith(prefix):
                candidates.append(split_dir / f"{sample_id}.npy")
                candidates.append(split_dir / f"{sample_id}.npz")
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f"Could not resolve raw IQ path for {sample_id} under {self.raw_root}")


@TRANSFORMS.register_module()
class LoadRawIQMemmapFromCOCOStem(BaseTransform):
    """Load raw complex IQ from split memmaps using the COCO sample index."""

    def __init__(
        self,
        memmap_root: str,
        *,
        target_shape: tuple[int, int] = (512, 512),
        to_float32: bool = True,
    ) -> None:
        self.memmap_root = Path(memmap_root)
        self.target_shape = tuple(int(v) for v in target_shape)
        self.to_float32 = to_float32

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        filename = Path(results["img_path"])
        split = _split_from_img_path(filename)
        index = _sample_index_from_path(filename)
        mmap = _open_memmap(self.memmap_root / f"{split}.npy")
        raw = np.asarray(mmap[index])
        if np.iscomplexobj(raw):
            iq = raw.astype(np.complex64, copy=False)
            pair = np.stack([iq.real, iq.imag], axis=-1)
        else:
            pair = raw.astype(np.float32, copy=False)
            if pair.ndim == 2 and pair.shape[0] == 2:
                pair = np.transpose(pair, (1, 0))
            if pair.ndim != 2 or pair.shape[-1] != 2:
                raise ValueError(f"Expected raw IQ pair [N,2] or complex [N], got {raw.shape}")
        if self.to_float32:
            pair = pair.astype(np.float32, copy=False)
        image = pair[:, None, :]
        results["img"] = image
        results["img_shape"] = self.target_shape
        results["ori_shape"] = self.target_shape
        results["scale_factor"] = (1.0, 1.0)
        results["raw_iq_shape"] = (2, int(pair.shape[0]))
        results["memmap_index"] = int(index)
        return results
        searched = ", ".join(str(path) for path in candidates[:8])
        raise FileNotFoundError(f"Raw IQ file not found for {stem_path}; searched: {searched}")


@MODELS.register_module()
class ComplexStftDetDataPreprocessor(DetDataPreprocessor):
    """MMDetection preprocessor that accepts two real/imag STFT channels."""

    def __init__(self, mean=None, std=None, **kwargs) -> None:
        super().__init__(mean=None, std=None, bgr_to_rgb=False, rgb_to_bgr=False, **kwargs)
        if mean is not None or std is not None:
            if mean is None or std is None:
                raise ValueError("mean and std must be provided together")
            if len(mean) != len(std):
                raise ValueError(f"STFT mean/std length mismatch: {len(mean)} and {len(std)}")
            if len(mean) < 1:
                raise ValueError("STFT mean/std must contain at least one channel")
            self._enable_normalize = True
            self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1), False)
            self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False


@MODELS.register_module()
class RawIQFilterbankDetDataPreprocessor(DetDataPreprocessor):
    """Generate detector-ready TF tensors from raw IQ inside the model graph."""

    def __init__(
        self,
        *,
        mean=None,
        std=None,
        num_bins: int = 512,
        kernel_size: int = 513,
        stride: int = 512,
        filterbank_kernel_sizes: tuple[int, ...] | list[int] | None = None,
        filterbank_strides: tuple[int, ...] | list[int] | None = None,
        filterbank_fusion: str = "single",
        filterbank_scale_logits_init: tuple[float, ...] | list[float] | None = None,
        filterbank_init: str = "fourier",
        filterbank_window: str = "blackman-harris",
        filterbank_residual_scale: float = 0.0,
        filterbank_residual_init_std: float = 0.0,
        filterbank_refiner_layers: int = 0,
        filterbank_refiner_kernel_size: int = 17,
        filterbank_refiner_dilations: tuple[int, ...] | list[int] | None = None,
        filterbank_refiner_residual_scale: float = 0.1,
        filterbank_refiner_init_std: float = 0.0,
        filterbank_type: str = "gabor",
        channel_mode: str = "realimag_logmag",
        trainable_filterbank: bool = True,
        output_size: tuple[int, int] | None = None,
        eps: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__(mean=None, std=None, bgr_to_rgb=False, rgb_to_bgr=False, **kwargs)
        self.output_size = tuple(int(v) for v in output_size) if output_size is not None else None
        self.filterbank_type = str(filterbank_type).lower().replace("_", "-")
        kernel_sizes = tuple(int(v) for v in (filterbank_kernel_sizes or ()))
        if self.filterbank_type in {"temporal-conv", "complex-temporal-conv", "ctf"}:
            if kernel_sizes:
                raise ValueError("temporal_conv filterbank does not support multiscale kernel sizes.")
            if int(stride) != 1:
                raise ValueError("temporal_conv filterbank is length-preserving and requires stride=1.")
            self.filterbank = ComplexTemporalConvFilterbank(
                num_bins=int(num_bins),
                kernel_size=int(kernel_size),
                init=filterbank_init,
                window=filterbank_window,
                residual_scale=float(filterbank_residual_scale),
                residual_init_std=float(filterbank_residual_init_std),
                refiner_layers=int(filterbank_refiner_layers),
                refiner_kernel_size=int(filterbank_refiner_kernel_size),
                refiner_dilations=tuple(int(v) for v in (filterbank_refiner_dilations or (1,))),
                refiner_residual_scale=float(filterbank_refiner_residual_scale),
                refiner_init_std=float(filterbank_refiner_init_std),
            )
        elif kernel_sizes:
            strides = tuple(int(v) for v in (filterbank_strides or ()))
            target_time_bins = int(self.output_size[1]) if self.output_size is not None else 0
            self.filterbank = MultiScaleComplexGaborFilterbank(
                num_bins=int(num_bins),
                kernel_sizes=kernel_sizes,
                strides=strides,
                target_time_bins=target_time_bins,
                fusion=str(filterbank_fusion),
                scale_logits_init=tuple(float(v) for v in (filterbank_scale_logits_init or ())),
                init=filterbank_init,
                window=filterbank_window,
                residual_scale=float(filterbank_residual_scale),
                residual_init_std=float(filterbank_residual_init_std),
            )
        else:
            if self.filterbank_type not in {"gabor", "fourier", "single"}:
                raise ValueError(f"Unsupported filterbank_type: {filterbank_type!r}")
            self.filterbank = ComplexGaborFilterbank(
                num_bins=int(num_bins),
                kernel_size=int(kernel_size),
                stride=int(stride),
                init=filterbank_init,
                window=filterbank_window,
                residual_scale=float(filterbank_residual_scale),
                residual_init_std=float(filterbank_residual_init_std),
            )
        if not trainable_filterbank:
            for param in self.filterbank.parameters():
                param.requires_grad_(False)
        self.channel_mode = str(channel_mode)
        if self.channel_mode not in {"realimag", "realimag_logmag", "logmag2ch"}:
            raise ValueError(f"Unsupported channel_mode: {self.channel_mode!r}")
        self.eps = float(eps)

        if mean is not None or std is not None:
            if mean is None or std is None:
                raise ValueError("mean and std must be provided together")
            if len(mean) != len(std):
                raise ValueError(f"Raw-IQ filterbank mean/std length mismatch: {len(mean)} and {len(std)}")
            self._enable_normalize = True
            self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1), False)
            self.register_buffer("std", torch.tensor(std, dtype=torch.float32).clamp_min(1e-6).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False

    @staticmethod
    def _packed_raw_to_pair(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            if x.shape[0] == 2:
                return x
            if x.shape[-1] == 2:
                return x.transpose(0, 1).contiguous()
        if x.ndim == 3:
            if x.shape[0] == 2:
                if x.shape[-1] == 1:
                    return x[..., 0]
                if x.shape[1] == 1:
                    return x[:, 0, :]
            if x.shape[-1] == 2:
                flat = x.reshape(-1, 2)
                return flat.transpose(0, 1).contiguous()
        raise ValueError(f"Expected packed raw IQ with two channels, got {tuple(x.shape)}")

    def _build_tf_tensor(self, raw_pairs: torch.Tensor) -> torch.Tensor:
        tokens = self.filterbank(raw_pairs)
        if tokens.ndim == 4:
            # [B, S, F, T] -> flatten scale into detector channels while
            # preserving the frequency-axis flip used by STFT/COCO tensors.
            spec = torch.flip(tokens, dims=(2,))
            real = spec.real.float()
            imag = spec.imag.float()
            if self.channel_mode == "realimag":
                out = torch.cat([real, imag], dim=1)
            elif self.channel_mode == "realimag_logmag":
                logmag = torch.log1p(torch.abs(spec).float().clamp_min(self.eps))
                out = torch.cat([real, imag, logmag], dim=1)
            else:
                logmag = torch.log1p(torch.abs(spec).float().clamp_min(self.eps))
                out = torch.cat([logmag, logmag], dim=1)
        else:
            spec = torch.flip(tokens, dims=(1,))
            real = spec.real.float()
            imag = spec.imag.float()
            if self.channel_mode == "realimag":
                out = torch.stack([real, imag], dim=1)
            elif self.channel_mode == "realimag_logmag":
                logmag = torch.log1p(torch.abs(spec).float().clamp_min(self.eps))
                out = torch.stack([real, imag, logmag], dim=1)
            else:
                logmag = torch.log1p(torch.abs(spec).float().clamp_min(self.eps))
                out = torch.stack([logmag, logmag], dim=1)
        if self.output_size is not None and tuple(out.shape[-2:]) != self.output_size:
            out = F.interpolate(out, size=self.output_size, mode="bilinear", align_corners=False)
        if self._enable_normalize:
            out = (out - self.mean) / self.std
        return out

    def _stack_raw_inputs(self, inputs: Any) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            if inputs.ndim == 4:
                pairs = [self._packed_raw_to_pair(item) for item in inputs]
            elif inputs.ndim == 3:
                pairs = [self._packed_raw_to_pair(inputs)]
            else:
                raise ValueError(f"Unsupported raw IQ batch tensor shape: {tuple(inputs.shape)}")
        else:
            pairs = [self._packed_raw_to_pair(item) for item in inputs]
        lengths = {int(pair.shape[-1]) for pair in pairs}
        if len(lengths) != 1:
            raise ValueError(f"Raw IQ batch contains variable lengths: {sorted(lengths)}")
        return torch.stack([pair.float() for pair in pairs], dim=0)

    def forward(self, data: dict, training: bool = False) -> dict:
        data = self.cast_data(data)
        raw_pairs = self._stack_raw_inputs(data["inputs"])
        inputs = self._build_tf_tensor(raw_pairs)
        data_samples = data.get("data_samples", None)

        if data_samples is not None:
            batch_input_shape = tuple(inputs.shape[-2:])
            for data_sample in data_samples:
                data_sample.set_metainfo(
                    {
                        "batch_input_shape": batch_input_shape,
                        "pad_shape": batch_input_shape,
                        "img_shape": batch_input_shape,
                    }
                )
            if self.boxtype2tensor:
                _samplelist_boxtype2tensor(data_samples)

        if training and self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)

        return {"inputs": inputs, "data_samples": data_samples}


@MODELS.register_module()
class PySigmoidFocalLoss(nn.Module):
    """Pure PyTorch sigmoid focal loss for mmcv-lite Windows smoke runs."""

    def __init__(
        self,
        use_sigmoid: bool = True,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if not use_sigmoid:
            raise ValueError("PySigmoidFocalLoss only supports sigmoid focal loss")
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
        avg_factor: float | None = None,
        reduction_override: str | None = None,
    ) -> torch.Tensor:
        reduction = reduction_override or self.reduction
        if pred.dim() != target.dim():
            target = F.one_hot(target, num_classes=pred.size(1) + 1)[:, : pred.size(1)]
        target = target.to(dtype=pred.dtype)
        pred_sigmoid = pred.sigmoid()
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * pt.pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none") * focal_weight
        if weight is not None:
            loss = loss * weight
        if reduction == "sum":
            loss = loss.sum()
        elif reduction == "mean":
            if avg_factor is not None:
                loss = loss.sum() / max(float(avg_factor), 1e-12)
            else:
                loss = loss.mean()
        elif reduction == "none":
            pass
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")
        return self.loss_weight * loss


@MODELS.register_module()
class ComplexStftCSPNeXt(CSPNeXt):
    """CSPNeXt stem variant for two-channel complex STFT tensors."""

    def __init__(self, input_channels: int = 2, norm_cfg=None, act_cfg=None, **kwargs) -> None:
        if norm_cfg is None:
            norm_cfg = dict(type="BN", momentum=0.03, eps=0.001)
        if act_cfg is None:
            act_cfg = dict(type="SiLU")
        super().__init__(norm_cfg=norm_cfg, act_cfg=act_cfg, **kwargs)
        first = self.stem[0]
        self.stem[0] = ConvModule(
            input_channels,
            first.conv.out_channels,
            first.conv.kernel_size,
            stride=first.conv.stride,
            padding=first.conv.padding,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )


def _complex_power_augmented(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if x.size(1) != 2:
        raise ValueError(f"Expected a two-channel real/imag tensor, got {tuple(x.shape)}")
    magnitude = torch.sqrt(torch.square(x[:, 0:1]) + torch.square(x[:, 1:2]) + eps)
    log_magnitude = torch.log1p(magnitude)
    return torch.cat([x, log_magnitude], dim=1)


@MODELS.register_module()
class ComplexPowerCSPNeXt(CSPNeXt):
    """CSPNeXt with a deterministic magnitude channel from real/imag STFT.

    The dataset input remains the WBSig-style two-channel complex spectrogram.
    The extra channel is computed inside the model so the baseline can test
    whether a complex-aware stem closes the gap to log-power image detectors.
    """

    def __init__(self, norm_cfg=None, act_cfg=None, **kwargs) -> None:
        if norm_cfg is None:
            norm_cfg = dict(type="BN", momentum=0.03, eps=0.001)
        if act_cfg is None:
            act_cfg = dict(type="SiLU")
        kwargs.pop("input_channels", None)
        super().__init__(norm_cfg=norm_cfg, act_cfg=act_cfg, **kwargs)
        first = self.stem[0]
        self.stem[0] = ConvModule(
            3,
            first.conv.out_channels,
            first.conv.kernel_size,
            stride=first.conv.stride,
            padding=first.conv.padding,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return super().forward(_complex_power_augmented(x))


@MODELS.register_module()
class ComplexStftCSPDarknet(CSPDarknet):
    """YOLOX CSPDarknet stem variant for two-channel STFT tensors."""

    def __init__(self, input_channels: int = 2, norm_cfg=None, act_cfg=None, **kwargs) -> None:
        if norm_cfg is None:
            norm_cfg = dict(type="BN", momentum=0.03, eps=0.001)
        if act_cfg is None:
            act_cfg = dict(type="Swish")
        super().__init__(norm_cfg=norm_cfg, act_cfg=act_cfg, **kwargs)
        first = self.stem.conv
        self.stem = Focus(
            input_channels,
            first.out_channels,
            kernel_size=first.kernel_size[0],
            stride=first.stride[0],
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )


@MODELS.register_module()
class ComplexPowerCSPDarknet(CSPDarknet):
    """YOLOX CSPDarknet with an internal magnitude channel from real/imag STFT."""

    def __init__(self, norm_cfg=None, act_cfg=None, **kwargs) -> None:
        if norm_cfg is None:
            norm_cfg = dict(type="BN", momentum=0.03, eps=0.001)
        if act_cfg is None:
            act_cfg = dict(type="Swish")
        kwargs.pop("input_channels", None)
        super().__init__(norm_cfg=norm_cfg, act_cfg=act_cfg, **kwargs)
        first = self.stem.conv
        self.stem = Focus(
            3,
            first.out_channels,
            kernel_size=first.kernel_size[0],
            stride=first.stride[0],
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return super().forward(_complex_power_augmented(x))


@MODELS.register_module()
class RawIQPassThroughDetDataPreprocessor(DetDataPreprocessor):
    """Stack raw IQ tensors while letting the model backbone do RF processing.

    The detector sees raw IQ as ``[B, 2, N]`` and the custom backbone emits
    image-shaped feature maps for MMDetection heads. Metadata still carries the
    virtual detector canvas size used by COCO boxes.
    """

    def __init__(self, *, detector_input_shape: tuple[int, int] = (512, 512), **kwargs) -> None:
        # The harness may inject STFT3 mean/std into the cfg; a raw-IQ pass-through never normalizes,
        # so drop them to avoid a duplicate-kwarg clash with the explicit mean=None/std=None below.
        kwargs.pop("mean", None)
        kwargs.pop("std", None)
        super().__init__(mean=None, std=None, bgr_to_rgb=False, rgb_to_bgr=False, **kwargs)
        self.detector_input_shape = tuple(int(v) for v in detector_input_shape)
        if len(self.detector_input_shape) != 2:
            raise ValueError("detector_input_shape must be (height, width)")

    @staticmethod
    def _packed_raw_to_pair(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            if x.shape[0] == 2:
                return x
            if x.shape[-1] == 2:
                return x.transpose(0, 1).contiguous()
        if x.ndim == 3:
            if x.shape[0] == 2:
                if x.shape[-1] == 1:
                    return x[..., 0]
                if x.shape[1] == 1:
                    return x[:, 0, :]
            if x.shape[-1] == 2:
                flat = x.reshape(-1, 2)
                return flat.transpose(0, 1).contiguous()
        raise ValueError(f"Expected packed raw IQ with two channels, got {tuple(x.shape)}")

    def _stack_raw_inputs(self, inputs: Any) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            if inputs.ndim in (3, 4):
                pairs = [self._packed_raw_to_pair(item) for item in inputs]
            else:
                raise ValueError(f"Unsupported raw IQ batch tensor shape: {tuple(inputs.shape)}")
        else:
            pairs = [self._packed_raw_to_pair(item) for item in inputs]
        lengths = {int(pair.shape[-1]) for pair in pairs}
        if len(lengths) != 1:
            raise ValueError(f"Raw IQ batch contains variable lengths: {sorted(lengths)}")
        return torch.stack([pair.float() for pair in pairs], dim=0)

    def forward(self, data: dict, training: bool = False) -> dict:
        data = self.cast_data(data)
        inputs = self._stack_raw_inputs(data["inputs"])
        data_samples = data.get("data_samples", None)
        if data_samples is not None:
            batch_input_shape = self.detector_input_shape
            for data_sample in data_samples:
                data_sample.set_metainfo(
                    {
                        "batch_input_shape": batch_input_shape,
                        "pad_shape": batch_input_shape,
                        "img_shape": batch_input_shape,
                    }
                )
            if self.boxtype2tensor:
                _samplelist_boxtype2tensor(data_samples)
        if training and self.batch_augments is not None:
            raise RuntimeError("RawIQPassThroughDetDataPreprocessor does not support image batch_augments.")
        return {"inputs": inputs, "data_samples": data_samples}


@MODELS.register_module()
class ComplexIQ1DPyramidBackbone(nn.Module):
    """Complex-valued 1D CNN on raw IQ + in-graph FFT bridge to a 2D vision neck.

    Raw IQ ``[B, 2, N]`` -> complex-1D stem/stages (downsample + widen) -> 3 complex
    feature taps ``[B, C_i, L_i]`` (each L_i a perfect square) -> per tap: reshape to
    ``[B, C_i, sqrt(L_i), sqrt(L_i)]``, FFT along the within-block time axis, then
    concat ``[real, imag, abs]`` on channels -> 3 REAL maps ``[B, 3*C_i, H_i, W_i]``
    for the CSPNeXtPAFPN neck (strides 8/16/32 on a 512 canvas).

    All complex ops are 1D (3D tensors); the FFT bridge + real/imag/abs projection
    are done manually because iq_pair_to_complex/ComplexToRealProjection require 3D.
    """

    def __init__(
        self,
        input_channels: int = 2,
        complex_channels: tuple[int, ...] = (64, 128, 256),
        out_lengths: tuple[int, ...] = (4096, 1024, 256),
        stem_kernel: int = 129,
        stem_stride: int = 64,
        stage_kernel: int = 7,
        stage_stride: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_channels != 2:
            raise ValueError("ComplexIQ1DPyramidBackbone expects 2 real IQ channels (1 complex).")
        if len(complex_channels) != 3 or len(out_lengths) != 3:
            raise ValueError("complex_channels and out_lengths must each have 3 entries.")
        for length in out_lengths:
            side = int(round(length ** 0.5))
            if side * side != length:
                raise ValueError(f"out_length {length} is not a perfect square.")
        self.complex_channels = tuple(int(c) for c in complex_channels)
        self.out_lengths = tuple(int(length) for length in out_lengths)
        c0, c1, c2 = self.complex_channels
        self.stem = nn.Sequential(
            NativeComplexConv1d(1, c0, kernel_size=stem_kernel, stride=stem_stride, padding=stem_kernel // 2),
            ComplexNormAct(c0, dropout=dropout),
        )
        self.stage2 = nn.Sequential(
            NativeComplexConv1d(c0, c1, kernel_size=stage_kernel, stride=stage_stride, padding=stage_kernel // 2),
            ComplexNormAct(c1, dropout=dropout),
        )
        self.stage3 = nn.Sequential(
            NativeComplexConv1d(c1, c2, kernel_size=stage_kernel, stride=stage_stride, padding=stage_kernel // 2),
            ComplexNormAct(c2, dropout=dropout),
        )

    @staticmethod
    def _fft_bridge(z: torch.Tensor, length: int) -> torch.Tensor:
        # z: complex [B, C, L]; reshape to square, FFT along within-block time, concat real/imag/abs.
        b, c, cur = z.shape
        side = int(round(length ** 0.5))
        if cur > length:
            z = z[..., :length]
        elif cur < length:
            z = F.pad(z, (0, length - cur))
        z = z.reshape(b, c, side, side)
        z = torch.fft.fft(z, dim=-1, norm="ortho")
        return torch.cat([z.real, z.imag, z.abs()], dim=1).contiguous()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        t1 = self.stem(x)       # complex [B, c0, ~4096]
        t2 = self.stage2(t1)    # complex [B, c1, ~1024]
        t3 = self.stage3(t2)    # complex [B, c2, ~256]
        f1 = self._fft_bridge(t1, self.out_lengths[0])  # [B, 3*c0, 64, 64]
        f2 = self._fft_bridge(t2, self.out_lengths[1])  # [B, 3*c1, 32, 32]
        f3 = self._fft_bridge(t3, self.out_lengths[2])  # [B, 3*c2, 16, 16]
        return (f1, f2, f3)
