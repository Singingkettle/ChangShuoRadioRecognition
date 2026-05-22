#!/usr/bin/env python3
"""Validate clean-paired synthetic AWGN AMC data."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_json(path):
    with Path(path).open("r") as f:
        return json.load(f)


def as_complex(iq):
    iq = np.asarray(iq, dtype=np.float64)
    if iq.shape[0] != 2:
        raise ValueError(f"Expected IQ shape (2, L), got {iq.shape}")
    return iq[0] + 1j * iq[1]


def estimate_snr_db(clean, noisy):
    noise = noisy - clean
    return 10 * np.log10(np.mean(np.abs(clean) ** 2) / (np.mean(np.abs(noise) ** 2) + 1e-12))


def validate(root, sample_limit, snr_tolerance_db):
    root = Path(root)
    split_payloads = {split: load_json(root / f"{split}.json") for split in ("train", "validation", "test")}
    clean_ids_by_split = {}
    sample_idx = set()
    counts = Counter()
    snr_errors = []
    examples = []

    for split, payload in split_payloads.items():
        data_list = payload["data_list"]
        clean_ids = {int(item["clean_id"]) for item in data_list}
        clean_ids_by_split[split] = clean_ids
        for item in data_list:
            idx = int(item["sample_idx"])
            if idx in sample_idx:
                raise ValueError(f"Duplicate sample_idx detected: {idx}")
            sample_idx.add(idx)
            counts[(split, item["modulation"], int(item["snr"]))] += 1
            for required in ("file_name", "clean_file_name", "clean_id", "sample_idx", "snr", "modulation"):
                if required not in item:
                    raise ValueError(f"{split} item missing required key {required}: {item}")

    for left in clean_ids_by_split:
        for right in clean_ids_by_split:
            if left >= right:
                continue
            overlap = clean_ids_by_split[left] & clean_ids_by_split[right]
            if overlap:
                raise ValueError(f"clean_id leakage between {left} and {right}: {sorted(overlap)[:5]}")

    all_items = []
    for split, payload in split_payloads.items():
        for item in payload["data_list"]:
            all_items.append((split, item))
    if sample_limit and sample_limit < len(all_items):
        rng = np.random.default_rng(0)
        choices = rng.choice(len(all_items), size=sample_limit, replace=False)
        items_to_check = [all_items[i] for i in choices]
    else:
        items_to_check = all_items

    for split, item in items_to_check:
        clean_path = root / "clean" / item["clean_file_name"]
        noisy_path = root / "iq" / item["file_name"]
        if not clean_path.exists() or not noisy_path.exists():
            raise FileNotFoundError(f"Missing clean/noisy file for item {item}")
        clean = as_complex(np.load(clean_path))
        noisy = as_complex(np.load(noisy_path))
        if clean.shape != noisy.shape:
            raise ValueError(f"Shape mismatch for {item}: clean {clean.shape}, noisy {noisy.shape}")
        measured = estimate_snr_db(clean, noisy)
        error = float(measured - float(item["snr"]))
        snr_errors.append(error)
        if abs(error) > snr_tolerance_db:
            raise ValueError(
                f"SNR error {error:.3f} dB exceeds tolerance for sample_idx={item['sample_idx']} "
                f"target={item['snr']} measured={measured:.3f}")
        if len(examples) < 5:
            examples.append({
                "split": split,
                "sample_idx": int(item["sample_idx"]),
                "modulation": item["modulation"],
                "target_snr": float(item["snr"]),
                "measured_snr": float(measured),
            })

    split_counts = defaultdict(int)
    for split, payload in split_payloads.items():
        split_counts[split] = len(payload["data_list"])

    report = {
        "root": str(root),
        "splits": dict(split_counts),
        "total_samples": int(sum(split_counts.values())),
        "total_clean_ids": {k: len(v) for k, v in clean_ids_by_split.items()},
        "count_rows": len(counts),
        "checked_samples": len(items_to_check),
        "mean_abs_snr_error_db": float(np.mean(np.abs(snr_errors))) if snr_errors else 0.0,
        "max_abs_snr_error_db": float(np.max(np.abs(snr_errors))) if snr_errors else 0.0,
        "examples": examples,
    }
    out = root / "validation_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--sample-limit", type=int, default=1000)
    parser.add_argument("--snr-tolerance-db", type=float, default=1.25)
    return parser.parse_args()


def main():
    args = parse_args()
    validate(args.root, args.sample_limit, args.snr_tolerance_db)


if __name__ == "__main__":
    main()
