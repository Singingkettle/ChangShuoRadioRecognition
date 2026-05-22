#!/usr/bin/env python3
"""Python fallback for the clean-paired synthetic AWGN AMC dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MODULATIONS = [
    "8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK",
    "4PAM", "16QAM", "64QAM", "QPSK", "WBFM",
]
DEFAULT_SNRS = list(range(-20, 20, 2))


def psk_symbols(rng, order, n):
    k = rng.integers(0, order, size=n)
    return np.exp(1j * (2 * np.pi * k / order + np.pi / order))


def qam_symbols(rng, side, n):
    levels = np.arange(-(side - 1), side, 2)
    x = rng.choice(levels, size=n) + 1j * rng.choice(levels, size=n)
    return x / np.sqrt(np.mean(np.abs(x) ** 2))


def message_signal(rng, n):
    w = rng.normal(size=n + 16)
    kernel = np.ones(9) / 9
    m = np.convolve(w, kernel, mode="same")[8:8 + n]
    return m / (np.max(np.abs(m)) + 1e-12)


def analytic_signal(x):
    n = x.size
    X = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = 1
        h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2
    return np.fft.ifft(X * h)


def fsk_like(rng, n, smooth=False):
    symbols = 2 * rng.integers(0, 2, size=n) - 1
    if smooth:
        xs = np.arange(-4, 5)
        g = np.exp(-(xs ** 2) / 5)
        g = g / g.sum()
        symbols = np.convolve(symbols, g, mode="same")
    phase = np.cumsum(0.55 * np.pi * symbols)
    return np.exp(1j * phase)


def synth_modulation(rng, modulation, frame_len):
    if modulation == "BPSK":
        x = psk_symbols(rng, 2, frame_len)
    elif modulation == "QPSK":
        x = psk_symbols(rng, 4, frame_len)
    elif modulation == "8PSK":
        x = psk_symbols(rng, 8, frame_len)
    elif modulation == "4PAM":
        x = rng.choice(np.array([-3, -1, 1, 3]) / np.sqrt(5), size=frame_len)
    elif modulation == "16QAM":
        x = qam_symbols(rng, 4, frame_len)
    elif modulation == "64QAM":
        x = qam_symbols(rng, 8, frame_len)
    elif modulation == "CPFSK":
        x = fsk_like(rng, frame_len, smooth=False)
    elif modulation == "GFSK":
        x = fsk_like(rng, frame_len, smooth=True)
    elif modulation == "AM-DSB":
        x = 1 + 0.65 * message_signal(rng, frame_len)
    elif modulation == "AM-SSB":
        x = analytic_signal(message_signal(rng, frame_len))
    elif modulation == "WBFM":
        x = np.exp(1j * 2.2 * np.cumsum(message_signal(rng, frame_len)))
    else:
        raise ValueError(f"Unsupported modulation: {modulation}")
    x = np.asarray(x, dtype=np.complex64).reshape(-1)[:frame_len]
    x = x - x.mean()
    if np.mean(np.abs(x) ** 2) < 1e-8:
        x = x + 1e-3 * rng.normal(size=x.shape)
    return x / np.sqrt(np.mean(np.abs(x) ** 2) + 1e-12)


def add_awgn(rng, clean, snr_db):
    signal_power = np.mean(np.abs(clean) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        rng.normal(size=clean.shape) + 1j * rng.normal(size=clean.shape)
    )
    return clean + noise


def iq_array(x):
    return np.stack([x.real, x.imag]).astype(np.float32)


def write_annotation(path, metainfo, data_list):
    path.write_text(json.dumps({"metainfo": metainfo, "data_list": data_list}, indent=2))


def generate(output_root, samples_per_class, frame_len, seed, snrs):
    rng = np.random.default_rng(seed)
    output_root = Path(output_root)
    (output_root / "iq").mkdir(parents=True, exist_ok=True)
    (output_root / "clean").mkdir(parents=True, exist_ok=True)

    splits = {"train": [], "validation": [], "test": []}
    clean_index = 0
    sample_index = 0
    snr_errors = []

    for modulation in MODULATIONS:
        order = rng.permutation(samples_per_class)
        train_ids = set(order[: int(0.70 * samples_per_class)])
        val_ids = set(order[int(0.70 * samples_per_class): int(0.85 * samples_per_class)])
        for local_id in range(samples_per_class):
            clean_id = clean_index
            clean = synth_modulation(rng, modulation, frame_len)
            clean_file = f"{clean_id:012d}.npy"
            np.save(output_root / "clean" / clean_file, iq_array(clean))
            if local_id in train_ids:
                split = "train"
            elif local_id in val_ids:
                split = "validation"
            else:
                split = "test"
            for snr_db in snrs:
                noisy = add_awgn(rng, clean, snr_db)
                file_name = f"{sample_index:012d}.npy"
                np.save(output_root / "iq" / file_name, iq_array(noisy))
                measured = 10 * np.log10(
                    np.mean(np.abs(clean) ** 2) /
                    (np.mean(np.abs(noisy - clean) ** 2) + 1e-12)
                )
                snr_errors.append(float(measured - snr_db))
                splits[split].append({
                    "file_name": file_name,
                    "clean_file_name": clean_file,
                    "clean_id": int(clean_id),
                    "sample_idx": int(sample_index),
                    "modulation": modulation,
                    "snr": int(snr_db),
                    "seed": int(seed),
                    "channel_type": "awgn",
                    "has_clean_signal": True,
                })
                sample_index += 1
            clean_index += 1

    metainfo = {
        "modulations": MODULATIONS,
        "snrs": [int(s) for s in snrs],
        "generator": "generate_python_fallback.py",
        "frame_len": int(frame_len),
        "samples_per_class": int(samples_per_class),
        "seed": int(seed),
        "channel_type": "awgn",
        "has_clean_signal": True,
    }
    for split, data_list in splits.items():
        write_annotation(output_root / f"{split}.json", metainfo, data_list)
    manifest = {
        **metainfo,
        "output_root": str(output_root),
        "total_clean": int(clean_index),
        "total_noisy": int(sample_index),
        "mean_snr_error_db": float(np.mean(np.abs(snr_errors))),
        "max_snr_error_db": float(np.max(np.abs(snr_errors))),
    }
    (output_root / "generator_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="/home/citybuster/Data/RCPS/processed/synthetic_awgn_amc_v1")
    parser.add_argument("--samples-per-class", type=int, default=1000)
    parser.add_argument("--frame-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--snrs", default=",".join(map(str, DEFAULT_SNRS)))
    return parser.parse_args()


def main():
    args = parse_args()
    snrs = [int(float(x)) for x in args.snrs.split(",") if x.strip()]
    generate(args.output_root, args.samples_per_class, args.frame_len, args.seed, snrs)


if __name__ == "__main__":
    main()
