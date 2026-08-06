# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""Self-contained complex-valued 1-D primitives and analytic filterbanks used
by the detection-side data preprocessors and the complex-1D backbone. Vendored
verbatim from the paper's iqdet package (complex_layers.py + the filterbank
classes from model.py) so the detection subproject needs no external iqdet."""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F

def iq_pair_to_complex(x: Tensor) -> Tensor:
    """Convert interleaved real IQ channels to a native complex tensor."""

    if torch.is_complex(x):
        if x.ndim != 3:
            raise ValueError("Complex IQ tensors must be shaped [batch, channels, samples].")
        return x.to(torch.complex64)
    if x.ndim != 3 or x.shape[1] % 2 != 0:
        raise ValueError("Real IQ tensors must be shaped [batch, 2 * complex_channels, samples].")
    real, imag = x.chunk(2, dim=1)
    return torch.complex(real, imag).to(torch.complex64)


def complex_to_iq_pair(x: Tensor) -> Tensor:
    """Convert a native complex tensor to concatenated real/imag channels."""

    if not torch.is_complex(x):
        return x
    return torch.cat([x.real, x.imag], dim=1)


class NativeComplexConv1d(nn.Module):
    """PyTorch-native complex Conv1d with complex-valued parameters."""

    def __init__(
        self,
        in_complex_channels: int,
        out_complex_channels: int,
        kernel_size: int,
        padding: int | None = None,
        stride: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_complex_channels,
            out_complex_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            dtype=torch.complex64,
        )

    @property
    def weight(self) -> Tensor:
        return self.conv.weight

    @property
    def bias(self) -> Tensor | None:
        return self.conv.bias

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(iq_pair_to_complex(x))


class ComplexRMSNorm1d(nn.Module):
    """RMS normalization for complex features using magnitude energy."""

    def __init__(self, complex_channels: int, eps: float = 1e-6, affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(complex_channels, dtype=torch.complex64))
            self.bias = nn.Parameter(torch.zeros(complex_channels, dtype=torch.complex64))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        z = iq_pair_to_complex(x)
        scale = z.abs().square().mean(dim=1, keepdim=True).add(self.eps).rsqrt()
        out = z * scale
        if self.affine and self.weight is not None and self.bias is not None:
            out = out * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        return out


class ComplexModReLU(nn.Module):
    """modReLU non-linearity for complex-valued activations."""

    def __init__(self, complex_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(complex_channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        z = iq_pair_to_complex(x)
        magnitude = z.abs()
        activated = torch.relu(magnitude + self.bias.view(1, -1, 1))
        return activated * z / magnitude.clamp_min(self.eps)


class ComplexDropout(nn.Module):
    """Dropout with one real-valued mask shared by real and imaginary parts."""

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        if p < 0.0 or p >= 1.0:
            raise ValueError("ComplexDropout requires 0 <= p < 1.")
        self.p = float(p)

    def forward(self, x: Tensor) -> Tensor:
        z = iq_pair_to_complex(x)
        if not self.training or self.p == 0.0:
            return z
        keep_prob = 1.0 - self.p
        mask = torch.empty_like(z.real).bernoulli_(keep_prob).div_(keep_prob)
        return z * mask


class ComplexNormAct(nn.Module):
    """Complex-native normalization, activation and dropout block."""

    def __init__(self, complex_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = ComplexRMSNorm1d(complex_channels)
        self.act = ComplexModReLU(complex_channels)
        self.dropout = ComplexDropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.act(self.norm(x)))


class ComplexDownsample1d(nn.Module):
    """Learned complex strided convolution used instead of unsupported pooling."""

    def __init__(self, complex_channels: int) -> None:
        super().__init__()
        self.down = NativeComplexConv1d(
            complex_channels,
            complex_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.post = ComplexNormAct(complex_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.post(self.down(x))


class ComplexToRealProjection(nn.Module):
    """Project complex features to real-valued task heads."""

    def __init__(self, mode: str = "real_imag_abs") -> None:
        super().__init__()
        valid_modes = {"real_imag", "real_imag_abs", "real_imag_abs_phase", "real_imag_abs_logabs"}
        if mode not in valid_modes:
            raise ValueError(f"projection mode must be one of {sorted(valid_modes)}, got {mode!r}.")
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        z = iq_pair_to_complex(x)
        parts = [z.real, z.imag]
        if self.mode in {"real_imag_abs", "real_imag_abs_phase", "real_imag_abs_logabs"}:
            parts.append(z.abs())
        if self.mode == "real_imag_abs_phase":
            parts.append(torch.atan2(z.imag, z.real))
        if self.mode == "real_imag_abs_logabs":
            parts.append(torch.log1p(z.abs()))
        return torch.cat(parts, dim=1)


def projection_multiplier(mode: str) -> int:
    if mode == "real_imag":
        return 2
    if mode == "real_imag_abs":
        return 3
    if mode == "real_imag_abs_phase":
        return 4
    if mode == "real_imag_abs_logabs":
        return 4
    raise ValueError(f"Unknown complex projection mode: {mode!r}")


# --------------------------- analytic filterbanks ---------------------------
class ComplexGaborFilterbank(nn.Module):
    """Learnable native-complex filterbank for raw-IQ time-frequency tokens.

    The default ``gabor`` initialization keeps the original broad Gaussian
    analysis filters. The ``fourier`` initialization uses evenly spaced complex
    Fourier atoms and a standard analysis window, which is useful when testing
    whether dense detection is bottlenecked by the learned TF front end rather
    than by the detector head.
    """

    def __init__(
        self,
        num_bins: int,
        kernel_size: int,
        stride: int,
        *,
        init: str = "gabor",
        window: str = "gaussian",
        residual_scale: float = 0.0,
        residual_init_std: float = 0.0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("filterbank_kernel_size must be odd for centred complex filters.")
        self.kernel_size = kernel_size
        self.stride = stride
        self.init = str(init).lower()
        self.window = str(window).lower().replace("_", "-")
        self.residual_scale = float(residual_scale)
        self.residual_init_std = float(residual_init_std)
        initial_weight = self._initial_kernels(num_bins, kernel_size, self.init, self.window)
        self.conv = None
        if self.residual_scale > 0.0:
            self.register_buffer("base_weight", initial_weight, persistent=True)
            self.residual_weight = nn.Parameter(torch.zeros_like(initial_weight))
            if self.residual_init_std > 0.0:
                with torch.no_grad():
                    noise_real = torch.randn_like(self.residual_weight.real) * self.residual_init_std
                    noise_imag = torch.randn_like(self.residual_weight.imag) * self.residual_init_std
                    self.residual_weight.copy_(torch.complex(noise_real, noise_imag))
        else:
            self.conv = nn.Conv1d(
                1,
                num_bins,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=stride,
                bias=False,
                dtype=torch.complex64,
            )
        with torch.no_grad():
            if self.conv is not None:
                self.conv.weight.copy_(initial_weight)

    @staticmethod
    def _analysis_window(window: str, num_bins: int, kernel_size: int) -> Tensor:
        dtype = torch.float32
        n = torch.arange(kernel_size, dtype=dtype)
        centered = n - (kernel_size - 1) / 2.0
        window = str(window).lower().replace("_", "-")
        if window == "gaussian":
            sigma = torch.full((num_bins,), kernel_size / 6.0, dtype=dtype)
            return torch.exp(-0.5 * (centered.unsqueeze(0) / sigma.unsqueeze(1)).pow(2))
        if window == "hann":
            base = torch.hann_window(kernel_size, periodic=False, dtype=dtype)
        elif window in {"blackman-harris", "blackmanharris"}:
            denom = max(float(kernel_size - 1), 1.0)
            base = (
                0.35875
                - 0.48829 * torch.cos(2.0 * torch.pi * n / denom)
                + 0.14128 * torch.cos(4.0 * torch.pi * n / denom)
                - 0.01168 * torch.cos(6.0 * torch.pi * n / denom)
            )
        elif window in {"rect", "rectangular", "boxcar"}:
            base = torch.ones(kernel_size, dtype=dtype)
        else:
            raise ValueError(f"Unsupported filterbank_window: {window!r}.")
        return base.unsqueeze(0).expand(num_bins, -1).contiguous()

    @staticmethod
    def _initial_kernels(num_bins: int, kernel_size: int, init: str = "gabor", window_name: str = "gaussian") -> Tensor:
        dtype = torch.float32
        n = torch.arange(kernel_size, dtype=dtype) - (kernel_size - 1) / 2.0
        init = str(init).lower().replace("_", "-")
        if init == "gabor":
            centers = torch.linspace(-0.45, 0.45, num_bins, dtype=dtype)
            analysis_window = ComplexGaborFilterbank._analysis_window(window_name, num_bins, kernel_size)
        elif init in {"fourier", "stft", "fft"}:
            centers = torch.arange(num_bins, dtype=dtype) / max(float(num_bins), 1.0) - 0.5
            analysis_window = ComplexGaborFilterbank._analysis_window(window_name, num_bins, kernel_size)
        else:
            raise ValueError(f"Unsupported filterbank_init: {init!r}.")
        # Conv1d is cross-correlation, so the analysis kernel needs the
        # negative complex exponential used by Fourier analysis.
        carrier = torch.exp(-1j * (2.0 * torch.pi) * centers.unsqueeze(1) * n.unsqueeze(0))
        kernels = analysis_window.to(torch.complex64) * carrier.to(torch.complex64)
        kernels = kernels / kernels.abs().square().sum(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
        return kernels.unsqueeze(1)

    def forward(self, x: Tensor) -> Tensor:
        z = iq_pair_to_complex(x)
        if self.conv is not None:
            return self.conv(z)
        weight = self.base_weight + self.residual_scale * self.residual_weight
        return F.conv1d(z, weight, padding=self.kernel_size // 2, stride=self.stride)


class ComplexTemporalConvFilterbank(nn.Module):
    """Length-preserving native-complex temporal Conv1d filterbank.

    This front end is intentionally simpler than the strided Gabor/Fourier
    filterbank used in the earlier raw-IQ baselines. It applies same-padded
    complex Conv1d filters with stride 1, so the IQ sample axis is preserved
    exactly. When initialized with Fourier or Gabor atoms, output channels keep
    a frequency-ordered interpretation and can still be consumed by the same
    time-frequency detector head.
    """

    def __init__(
        self,
        num_bins: int,
        kernel_size: int,
        *,
        init: str = "fourier",
        window: str = "hann",
        residual_scale: float = 0.0,
        residual_init_std: float = 0.0,
        refiner_layers: int = 0,
        refiner_kernel_size: int = 17,
        refiner_dilations: tuple[int, ...] = (1,),
        refiner_residual_scale: float = 0.1,
        refiner_init_std: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("temporal_conv kernel_size must be odd for length-preserving same padding.")
        if refiner_kernel_size % 2 == 0:
            raise ValueError("temporal_conv refiner_kernel_size must be odd for same padding.")
        if refiner_layers < 0:
            raise ValueError("temporal_conv refiner_layers must be non-negative.")
        self.num_bins = int(num_bins)
        self.kernel_size = int(kernel_size)
        self.stride = 1
        self.init = str(init).lower().replace("_", "-")
        self.window = str(window).lower().replace("_", "-")
        self.residual_scale = float(residual_scale)
        self.residual_init_std = float(residual_init_std)
        self.refiner_layers = int(refiner_layers)
        self.refiner_kernel_size = int(refiner_kernel_size)
        self.refiner_dilations = tuple(int(v) for v in refiner_dilations) or (1,)
        self.refiner_residual_scale = float(refiner_residual_scale)
        self.refiner_init_std = float(refiner_init_std)
        initial_weight = self._initial_kernels(self.num_bins, self.kernel_size, self.init, self.window)
        self.conv = None
        if self.residual_scale > 0.0:
            if bias:
                raise ValueError("residual temporal_conv filterbank does not support bias.")
            self.register_buffer("base_weight", initial_weight, persistent=True)
            self.residual_weight = nn.Parameter(torch.zeros_like(initial_weight))
            if self.residual_init_std > 0.0:
                with torch.no_grad():
                    noise_real = torch.randn_like(self.residual_weight.real) * self.residual_init_std
                    noise_imag = torch.randn_like(self.residual_weight.imag) * self.residual_init_std
                    self.residual_weight.copy_(torch.complex(noise_real, noise_imag))
        else:
            self.conv = nn.Conv1d(
                1,
                self.num_bins,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1,
                bias=bool(bias),
                dtype=torch.complex64,
            )
            with torch.no_grad():
                self.conv.weight.copy_(initial_weight)
                if self.conv.bias is not None:
                    self.conv.bias.zero_()
        self.refiner = (
            ComplexDepthwiseTemporalRefiner(
                channels=self.num_bins,
                layers=self.refiner_layers,
                kernel_size=self.refiner_kernel_size,
                dilations=self.refiner_dilations,
                residual_scale=self.refiner_residual_scale,
                init_std=self.refiner_init_std,
            )
            if self.refiner_layers > 0
            else nn.Identity()
        )

    @staticmethod
    def _identity_kernels(num_bins: int, kernel_size: int, init: str) -> Tensor:
        weight = torch.zeros(num_bins, 1, kernel_size, dtype=torch.complex64)
        center = kernel_size // 2
        phases = torch.linspace(-torch.pi, torch.pi, num_bins, dtype=torch.float32)
        weight[:, 0, center] = torch.exp(1j * phases).to(torch.complex64)
        if init in {"identity-noise", "delta-noise"}:
            noise_real = torch.randn(num_bins, 1, kernel_size, dtype=torch.float32) * 1e-3
            noise_imag = torch.randn(num_bins, 1, kernel_size, dtype=torch.float32) * 1e-3
            weight = weight + torch.complex(noise_real, noise_imag).to(torch.complex64)
        return weight

    @classmethod
    def _initial_kernels(cls, num_bins: int, kernel_size: int, init: str, window: str) -> Tensor:
        init = str(init).lower().replace("_", "-")
        if init in {"identity", "delta", "identity-noise", "delta-noise"}:
            return cls._identity_kernels(num_bins, kernel_size, init)
        return ComplexGaborFilterbank._initial_kernels(num_bins, kernel_size, init, window)

    def forward(self, x: Tensor) -> Tensor:
        z = iq_pair_to_complex(x)
        if self.conv is not None:
            out = self.conv(z)
        else:
            weight = self.base_weight + self.residual_scale * self.residual_weight
            out = F.conv1d(z, weight, padding=self.kernel_size // 2, stride=1)
        out = self.refiner(out)
        if out.shape[-1] != z.shape[-1]:
            raise RuntimeError(
                f"ComplexTemporalConvFilterbank changed sample length from {z.shape[-1]} to {out.shape[-1]}."
            )
        return out


class ComplexDepthwiseTemporalRefiner(nn.Module):
    """Length-preserving per-frequency complex temporal refinement.

    The module keeps the frequency-bin ordering produced by the CTF analysis
    filters: every bin receives its own grouped complex Conv1d along time and a
    small residual update. This gives the raw-IQ front end extra temporal
    modeling capacity without pooling, striding, or global channel mixing.
    """

    def __init__(
        self,
        channels: int,
        layers: int,
        kernel_size: int,
        dilations: tuple[int, ...],
        residual_scale: float,
        init_std: float = 0.0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("ComplexDepthwiseTemporalRefiner requires an odd kernel_size.")
        if layers < 1:
            raise ValueError("ComplexDepthwiseTemporalRefiner requires at least one layer.")
        if not dilations:
            dilations = (1,)
        self.residual_scale = float(residual_scale)
        self.blocks = nn.ModuleList()
        for layer_idx in range(int(layers)):
            dilation = int(dilations[layer_idx % len(dilations)])
            if dilation < 1:
                raise ValueError("ComplexDepthwiseTemporalRefiner dilations must be positive.")
            conv = nn.Conv1d(
                int(channels),
                int(channels),
                kernel_size=int(kernel_size),
                padding=dilation * (int(kernel_size) // 2),
                dilation=dilation,
                groups=int(channels),
                bias=False,
                dtype=torch.complex64,
            )
            with torch.no_grad():
                if init_std > 0.0:
                    real = torch.randn_like(conv.weight.real) * float(init_std)
                    imag = torch.randn_like(conv.weight.imag) * float(init_std)
                    conv.weight.copy_(torch.complex(real, imag))
                else:
                    conv.weight.zero_()
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "norm": ComplexRMSNorm1d(int(channels)),
                        "act": ComplexModReLU(int(channels)),
                        "conv": conv,
                    }
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        z = iq_pair_to_complex(x)
        out = z
        for block in self.blocks:
            residual = block["conv"](block["act"](block["norm"](out)))
            out = out + self.residual_scale * residual
        return out


class MultiScaleComplexGaborFilterbank(nn.Module):
    """Multi-resolution learnable complex filterbank for joint TF detection.

    The output keeps scale as a feature-channel dimension, shaped
    [batch, scales, frequency_bins, time_bins]. This lets the dense TF head
    compare multiple temporal resolutions without treating scale as frequency.
    """

    def __init__(
        self,
        num_bins: int,
        kernel_sizes: tuple[int, ...],
        strides: tuple[int, ...],
        target_time_bins: int = 0,
        fusion: str = "stack",
        scale_logits_init: tuple[float, ...] = (),
        init: str = "gabor",
        window: str = "gaussian",
        residual_scale: float = 0.0,
        residual_init_std: float = 0.0,
    ) -> None:
        super().__init__()
        if not kernel_sizes:
            raise ValueError("MultiScaleComplexGaborFilterbank requires at least one kernel size.")
        if strides and len(strides) != len(kernel_sizes):
            raise ValueError("filterbank_strides must match filterbank_kernel_sizes when provided.")
        if not strides:
            strides = tuple(max(1, int(kernel // 4)) for kernel in kernel_sizes)
        self.fusion = str(fusion).lower()
        if self.fusion not in {"stack", "gated_sum"}:
            raise ValueError(f"Unsupported filterbank scale fusion: {fusion!r}")
        self.target_time_bins = int(target_time_bins)
        self.branches = nn.ModuleList(
            ComplexGaborFilterbank(
                num_bins=num_bins,
                kernel_size=int(kernel),
                stride=int(stride),
                init=init,
                window=window,
                residual_scale=residual_scale,
                residual_init_std=residual_init_std,
            )
            for kernel, stride in zip(kernel_sizes, strides)
        )
        self.scale_logits = nn.Parameter(torch.zeros(len(kernel_sizes))) if self.fusion == "gated_sum" else None
        if self.scale_logits is not None and scale_logits_init:
            if len(scale_logits_init) != len(kernel_sizes):
                raise ValueError("filterbank_scale_logits_init must match filterbank_kernel_sizes.")
            with torch.no_grad():
                self.scale_logits.copy_(torch.tensor(scale_logits_init, dtype=self.scale_logits.dtype))

    @staticmethod
    def _interpolate_complex(x: Tensor, size: int) -> Tensor:
        if x.shape[-1] == int(size):
            return x
        pair = complex_to_iq_pair(x)
        pair = F.interpolate(pair, size=int(size), mode="linear", align_corners=False)
        return iq_pair_to_complex(pair)

    def forward(self, x: Tensor) -> Tensor:
        views = [branch(x) for branch in self.branches]
        target_time_bins = self.target_time_bins if self.target_time_bins > 0 else max(view.shape[-1] for view in views)
        aligned = [self._interpolate_complex(view, target_time_bins) for view in views]
        stacked = torch.stack(aligned, dim=1)
        if self.scale_logits is None:
            return stacked
        weights = torch.softmax(self.scale_logits, dim=0).to(dtype=stacked.real.dtype)
        return (stacked * weights.view(1, -1, 1, 1)).sum(dim=1)
