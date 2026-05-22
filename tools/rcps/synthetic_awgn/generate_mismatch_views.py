#!/usr/bin/env python3
"""Generate test-only mismatch views from clean-paired synthetic AMC data.

The original strict AWGN-DPC dataset uses clean signal s and AWGN views x_gamma.
This helper keeps the same clean_id split and modulation labels, but changes the
forward channel for test views. It is intended for stress-testing whether a
method trained on clean AWGN remains useful when the test degradation is not the
same ideal AWGN kernel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_SNRS = list(range(-20, 20, 2))


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def as_complex(iq: np.ndarray) -> np.ndarray:
    iq = np.asarray(iq, dtype=np.float32)
    if iq.shape[0] != 2:
        raise ValueError(f"Expected IQ shape (2, L), got {iq.shape}")
    return iq[0].astype(np.float64) + 1j * iq[1].astype(np.float64)


def iq_array(x: np.ndarray) -> np.ndarray:
    return np.stack([x.real, x.imag]).astype(np.float32)


def normalize_power(x: np.ndarray) -> np.ndarray:
    return x / np.sqrt(np.mean(np.abs(x) ** 2) + 1e-12)


def apply_impairment(rng: np.random.Generator, clean: np.ndarray, channel_type: str) -> np.ndarray:
    x = np.asarray(clean, dtype=np.complex128).copy()
    n = x.size
    t = np.arange(n, dtype=np.float64)

    if channel_type in {"awgn", "none"}:
        y = x
    elif channel_type == "phase":
        phi = rng.uniform(-np.pi, np.pi)
        y = x * np.exp(1j * phi)
    elif channel_type == "freq":
        freq = rng.uniform(-0.035, 0.035)
        phi = rng.uniform(-np.pi, np.pi)
        y = x * np.exp(1j * (2.0 * np.pi * freq * t + phi))
    elif channel_type == "multipath":
        delay = int(rng.integers(1, 5))
        tap_mag = rng.uniform(0.15, 0.55)
        tap_phase = rng.uniform(-np.pi, np.pi)
        delayed = np.zeros_like(x)
        delayed[delay:] = x[:-delay]
        y = x + tap_mag * np.exp(1j * tap_phase) * delayed
    elif channel_type == "phase_freq_multipath":
        freq = rng.uniform(-0.025, 0.025)
        phi = rng.uniform(-np.pi, np.pi)
        y = x * np.exp(1j * (2.0 * np.pi * freq * t + phi))
        delay = int(rng.integers(1, 5))
        tap_mag = rng.uniform(0.10, 0.35)
        tap_phase = rng.uniform(-np.pi, np.pi)
        delayed = np.zeros_like(y)
        delayed[delay:] = y[:-delay]
        y = y + tap_mag * np.exp(1j * tap_phase) * delayed
    else:
        raise ValueError(f"Unsupported channel_type: {channel_type}")

    return normalize_power(y)


def add_noise(rng: np.random.Generator, signal: np.ndarray, snr_db: float, noise_type: str) -> np.ndarray:
    signal_power = float(np.mean(np.abs(signal) ** 2))
    noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    if noise_type == "gaussian":
        noise = np.sqrt(noise_power / 2.0) * (rng.normal(size=signal.shape) + 1j * rng.normal(size=signal.shape))
    elif noise_type == "laplace":
        # Complex Laplace with variance matched to noise_power.
        scale = np.sqrt(noise_power / 4.0)
        noise = scale * (rng.laplace(size=signal.shape) + 1j * rng.laplace(size=signal.shape))
    elif noise_type == "mixture":
        base_power = 0.55 * noise_power
        burst_power = 8.0 * noise_power
        mask = rng.random(size=signal.shape) < 0.06
        noise = np.sqrt(base_power / 2.0) * (rng.normal(size=signal.shape) + 1j * rng.normal(size=signal.shape))
        burst = np.sqrt(burst_power / 2.0) * (rng.normal(size=signal.shape) + 1j * rng.normal(size=signal.shape))
        noise = noise + mask * burst
    else:
        raise ValueError(f"Unsupported noise_type: {noise_type}")
    return signal + noise


def unique_clean_items(test_rows: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for item in test_rows:
        cid = int(item["clean_id"])
        if cid in seen:
            continue
        seen.add(cid)
        out.append(item)
    return out


def generate(args: argparse.Namespace) -> None:
    source = Path(args.source_root)
    out_root = Path(args.output_root)
    (out_root / "iq").mkdir(parents=True, exist_ok=True)
    (out_root / "clean").mkdir(parents=True, exist_ok=True)

    src_test = load_json(source / "test.json")
    clean_rows = unique_clean_items(src_test["data_list"])
    if args.max_clean and args.max_clean > 0:
        clean_rows = clean_rows[: args.max_clean]
    snrs = [float(x) for x in args.snrs.split(",") if x.strip()]
    rng = np.random.default_rng(args.seed)

    data_list = []
    sample_idx = 0
    channel_label = f"{args.channel_type}_{args.noise_type}"
    impairment_powers = []

    for item in clean_rows:
        clean_file = item["clean_file_name"]
        clean = as_complex(np.load(source / "clean" / clean_file))
        clean = normalize_power(clean)
        out_clean = out_root / "clean" / clean_file
        if not out_clean.exists():
            np.save(out_clean, iq_array(clean))

        for snr_db in snrs:
            impaired = apply_impairment(rng, clean, args.channel_type)
            noisy = add_noise(rng, impaired, snr_db, args.noise_type)
            file_name = f"{sample_idx:012d}.npy"
            np.save(out_root / "iq" / file_name, iq_array(noisy))
            impairment = impaired - clean
            impairment_powers.append(float(np.mean(np.abs(impairment) ** 2)))
            data_list.append({
                "file_name": file_name,
                "clean_file_name": clean_file,
                "clean_id": int(item["clean_id"]),
                "sample_idx": int(sample_idx),
                "global_sample_idx": int(sample_idx),
                "modulation": item["modulation"],
                "snr": float(snr_db),
                "snr_label": float(snr_db),
                "seed": int(args.seed),
                "channel_type": channel_label,
                "base_channel_type": args.channel_type,
                "noise_type": args.noise_type,
                "has_clean_signal": True,
                "mismatch_test_only": True,
            })
            sample_idx += 1

    metainfo = dict(src_test.get("metainfo", {}))
    metainfo.update({
        "generator": "generate_mismatch_views.py",
        "source_root": str(source),
        "channel_type": channel_label,
        "base_channel_type": args.channel_type,
        "noise_type": args.noise_type,
        "snrs": snrs,
        "seed": int(args.seed),
        "test_only": True,
        "has_clean_signal": True,
    })
    payload = {"metainfo": metainfo, "data_list": data_list}
    write_json(out_root / "test.json", payload)
    write_json(out_root / "train.json", {"metainfo": metainfo, "data_list": []})
    write_json(out_root / "validation.json", {"metainfo": metainfo, "data_list": []})
    manifest = {
        "output_root": str(out_root),
        "source_root": str(source),
        "channel_type": channel_label,
        "clean_count": len(clean_rows),
        "total_samples": len(data_list),
        "snrs": snrs,
        "seed": int(args.seed),
        "mean_impairment_power": float(np.mean(impairment_powers)) if impairment_powers else 0.0,
        "max_impairment_power": float(np.max(impairment_powers)) if impairment_powers else 0.0,
    }
    write_json(out_root / "generator_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="/home/citybuster/Data/RCPS/processed/synthetic_awgn_amc_v1")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--channel-type", default="phase_freq_multipath", choices=["awgn", "phase", "freq", "multipath", "phase_freq_multipath"])
    parser.add_argument("--noise-type", default="gaussian", choices=["gaussian", "laplace", "mixture"])
    parser.add_argument("--snrs", default=",".join(str(x) for x in DEFAULT_SNRS))
    parser.add_argument("--seed", type=int, default=3026)
    parser.add_argument("--max-clean", type=int, default=0, help="Optional quick subset of clean test items.")
    return parser.parse_args()


def main() -> None:
    generate(parse_args())


if __name__ == "__main__":
    main()
