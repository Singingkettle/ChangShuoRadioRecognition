# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from prepare_torchsig_coco import (
    CANONICAL_CLASS_NAMES,
    apply_dataset_preset,
    categories_from_map,
    category_id_for_key,
    class_key,
    create_torchsig_dataset,
    jsonable,
    repo_root,
    sample_with_retries,
)


def parse_snr_buckets(text: str) -> list[tuple[float, float]]:
    buckets: list[tuple[float, float]] = []
    for chunk in text.replace(";", "|").split("|"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [float(part.strip()) for part in chunk.split(",") if part.strip()]
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                "SNR buckets must look like '-10,0;0,10;10,20'."
            )
        lo, hi = parts
        if hi <= lo:
            raise argparse.ArgumentTypeError("Each SNR bucket must satisfy high > low.")
        buckets.append((lo, hi))
    if not buckets:
        raise argparse.ArgumentTypeError("At least one SNR bucket is required.")
    return buckets


def install_fast_snr_update() -> None:
    """Use signal-power SNR scaling instead of TorchSig's spectrogram estimate.

    TorchSig's default update path computes an STFT for every generated signal
    to estimate SNR and occupied bandwidth. That is useful for precise metadata
    refinement, but it is prohibitively slow for 262k-sample, large-scale data
    generation. For stratified SNR experiments we keep TorchSig's nominal
    time-frequency metadata and set SNR by scaling the signal power relative to
    the configured noise floor.
    """

    import torchsig.datasets.datasets as torchsig_datasets

    def _fast_update_signal_snr_bandwidth(dataset: Any, new_signal: Any) -> None:
        snr_db = float(np.round(dataset.random_generator.uniform(new_signal.snr_db_min, new_signal.snr_db_max), 1))
        signal_power = float(np.mean(np.abs(np.asarray(new_signal.data)) ** 2))
        if not np.isfinite(signal_power) or signal_power <= 0.0:
            signal_power = 1.0
        noise_power_linear = 10.0 ** (float(dataset.noise_power_db) / 10.0)
        target_power = noise_power_linear * (10.0 ** (snr_db / 10.0))
        scale = np.sqrt(max(target_power, 1e-30) / max(signal_power, 1e-30))
        new_signal.data *= scale
        new_signal["snr_db"] = snr_db

    torchsig_datasets.update_signal_snr_bandwidth = _fast_update_signal_snr_bandwidth


def build_bucket_datasets(args: argparse.Namespace, split: str) -> list[Any]:
    split_offset = {"train": 0, "val": 1000, "test": 2000}[split]
    datasets: list[Any] = []
    for bucket_idx, (snr_min, snr_max) in enumerate(args.snr_buckets):
        bucket_args = copy.deepcopy(args)
        bucket_args.snr_db_min = float(snr_min)
        bucket_args.snr_db_max = float(snr_max)
        seed = args.seed + split_offset + bucket_idx * 100_000
        datasets.append(create_torchsig_dataset(bucket_args, seed=seed))
    return datasets


def write_split(
    split: str,
    count: int,
    args: argparse.Namespace,
    out_root: Path,
    category_map: dict[tuple[int, str], int],
    next_category_id: int,
) -> int:
    raw_dir = out_root / "raw" / split
    meta_dir = out_root / "metadata"
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    datasets = build_bucket_datasets(args, split)
    metadata_lines: list[str] = []
    retry_count = 0
    total_instances = 0
    bucket_sample_counts = [0 for _ in args.snr_buckets]
    bucket_instance_counts = [0 for _ in args.snr_buckets]

    for sample_idx in range(count):
        bucket_idx = sample_idx % len(args.snr_buckets)
        snr_min, snr_max = args.snr_buckets[bucket_idx]
        iq, instances, retries = sample_with_retries(
            datasets[bucket_idx],
            split=split,
            sample_idx=sample_idx,
            max_retries=args.max_sample_retries,
        )
        retry_count += retries
        sample_id = f"{split}_{sample_idx:06d}"
        npz_path = raw_dir / f"{sample_id}.npz"
        np.savez_compressed(npz_path, iq=np.asarray(iq, dtype=np.complex64))

        serial_instances: list[dict[str, Any]] = []
        sample_rate = float(instances[0].get("sample_rate", args.sample_rate)) if instances else args.sample_rate
        for inst in instances:
            key = class_key(inst)
            inst = dict(inst)
            cat_id, next_category_id = category_id_for_key(key, category_map, next_category_id)
            inst["category_id"] = cat_id
            inst["snr_bucket_min"] = float(snr_min)
            inst["snr_bucket_max"] = float(snr_max)
            serial_instances.append(jsonable(inst))

        bucket_sample_counts[bucket_idx] += 1
        bucket_instance_counts[bucket_idx] += len(serial_instances)
        total_instances += len(serial_instances)
        metadata_lines.append(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "raw_path": str(npz_path.as_posix()),
                    "num_iq_samples": int(len(iq)),
                    "sample_rate": sample_rate,
                    "snr_bucket": [float(snr_min), float(snr_max)],
                    "snr_bucket_index": int(bucket_idx),
                    "instances": serial_instances,
                },
                ensure_ascii=False,
            )
        )
        if (sample_idx + 1) % 100 == 0 or sample_idx + 1 == count:
            print(
                f"[prepare-iq-stratified] {split}: generated {sample_idx + 1}/{count} samples",
                flush=True,
            )

    (meta_dir / f"{split}.jsonl").write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")
    print(
        f"[prepare-iq-stratified] {split}: {count} raw IQ samples, {total_instances} instances, "
        f"{len(category_map)} categories, {retry_count} retries, "
        f"bucket_samples={bucket_sample_counts}, bucket_instances={bucket_instance_counts}",
        flush=True,
    )
    return next_category_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="data/torchsig_widesnr_iq")
    parser.add_argument(
        "--preset",
        choices=["custom", "wbsig53-paper", "wbsig53-clean-like", "torchsig-wideband-default"],
        default="custom",
    )
    parser.add_argument("--train", type=int, default=10_000)
    parser.add_argument("--val", type=int, default=2_000)
    parser.add_argument("--test", type=int, default=2_000)
    parser.add_argument("--num-iq-samples", type=int, default=262_144)
    parser.add_argument("--sample-rate", type=float, default=10_000_000.0)
    parser.add_argument("--num-signals-min", type=int, default=1)
    parser.add_argument("--num-signals-max", type=int, default=6)
    parser.add_argument("--impairment-level", type=int, default=2)
    parser.add_argument("--fft-size", type=int, default=512)
    parser.add_argument("--stft-fft", type=int, default=512)
    parser.add_argument("--stft-hop", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260640)
    parser.add_argument("--duration-min-frac", type=float, default=0.05)
    parser.add_argument("--duration-max-frac", type=float, default=1.0)
    parser.add_argument("--bandwidth-min-frac", type=float, default=0.0125)
    parser.add_argument("--bandwidth-max-frac", type=float, default=0.49)
    parser.add_argument("--center-freq-min-frac", type=float, default=-0.40)
    parser.add_argument("--center-freq-max-frac", type=float, default=0.40)
    parser.add_argument("--snr-db-min", type=float, default=None,
                        help="Lower edge of the SNR range (default -10). Samples are stratified "
                             "into --snr-num-buckets equal buckets across [min, max].")
    parser.add_argument("--snr-db-max", type=float, default=None,
                        help="Upper edge of the SNR range (default 40).")
    parser.add_argument(
        "--snr-buckets",
        type=parse_snr_buckets,
        default=None,
        help="Explicit buckets, for example '-20,-10;-10,0;0,10'. Mutually exclusive with "
             "--snr-db-min/--snr-db-max: pass either the range or the buckets, not both.",
    )
    parser.add_argument("--snr-num-buckets", type=int, default=3,
                        help="How many equal buckets to cut [--snr-db-min, --snr-db-max] into "
                             "when --snr-buckets is not given. The paper used 3.")
    parser.add_argument("--cochannel-overlap-probability", type=float, default=0.1)
    parser.add_argument("--noise-power-db", type=float, default=None)
    parser.add_argument("--max-sample-retries", type=int, default=64)
    parser.add_argument(
        "--fast-snr-update",
        action="store_true",
        help="Scale generated signal power to target SNR without TorchSig's per-signal STFT refinement.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_snr_stratification(args: argparse.Namespace) -> argparse.Namespace:
    """Reconcile --snr-db-min/--snr-db-max with --snr-buckets, loudly.

    An earlier version derived the range from the buckets unconditionally, so a command
    that passed ``--snr-db-min -20 --snr-db-max 10`` without also passing ``--snr-buckets``
    silently generated over the *default* bucket span instead. That is the difference
    between the benchmark this repository describes and a much easier one, with no error
    and no warning. The range and the buckets are now a single choice: give one or the
    other, never a conflicting pair.
    """
    explicit_range = args.snr_db_min is not None or args.snr_db_max is not None
    if args.snr_buckets is not None:
        bucket_lo = min(lo for lo, _ in args.snr_buckets)
        bucket_hi = max(hi for _, hi in args.snr_buckets)
        if explicit_range:
            lo = bucket_lo if args.snr_db_min is None else args.snr_db_min
            hi = bucket_hi if args.snr_db_max is None else args.snr_db_max
            if (lo, hi) != (bucket_lo, bucket_hi):
                raise SystemExit(
                    "[prepare-iq-stratified] --snr-db-min/--snr-db-max "
                    f"({lo}, {hi}) disagree with the span of --snr-buckets "
                    f"({bucket_lo}, {bucket_hi}). Pass the range or the buckets, not both."
                )
        args.snr_db_min, args.snr_db_max = bucket_lo, bucket_hi
    else:
        if args.snr_db_min is None:
            args.snr_db_min = -10.0
        if args.snr_db_max is None:
            args.snr_db_max = 40.0
        if args.snr_db_max <= args.snr_db_min:
            raise SystemExit("[prepare-iq-stratified] --snr-db-max must exceed --snr-db-min.")
        if args.snr_num_buckets < 1:
            raise SystemExit("[prepare-iq-stratified] --snr-num-buckets must be >= 1.")
        step = (args.snr_db_max - args.snr_db_min) / args.snr_num_buckets
        args.snr_buckets = [
            (args.snr_db_min + i * step, args.snr_db_min + (i + 1) * step)
            for i in range(args.snr_num_buckets)
        ]
    print(
        "[prepare-iq-stratified] SNR range "
        f"[{args.snr_db_min}, {args.snr_db_max}] dB in {len(args.snr_buckets)} buckets: "
        + "; ".join(f"{lo:g},{hi:g}" for lo, hi in args.snr_buckets),
        flush=True,
    )
    return args


def generation_provenance(args: argparse.Namespace) -> dict:
    """Record what actually produced this dataset, so drift is diagnosable later."""
    try:
        import torchsig

        torchsig_version = getattr(torchsig, "__version__", "unknown")
    except Exception:
        torchsig_version = "not-importable"
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:
        commit = "unknown"
    return {
        "torchsig_version": torchsig_version,
        "git_commit": commit,
        "seed": args.seed,
        "argv": sys.argv,
        "python": sys.version.split()[0],
    }


def main() -> None:
    args = resolve_snr_stratification(apply_dataset_preset(parse_args()))
    if args.fast_snr_update:
        install_fast_snr_update()

    out_root = repo_root() / args.out_root
    if out_root.exists() and args.force:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    category_map: dict[tuple[int, str], int] = {}
    next_category_id = len(CANONICAL_CLASS_NAMES)
    for split, count in (("train", args.train), ("val", args.val), ("test", args.test)):
        next_category_id = write_split(split, count, args, out_root, category_map, next_category_id)

    categories = categories_from_map(category_map)
    summary = {
        "out_root": str(out_root),
        "format": "raw_iq_jsonl_stratified_snr",
        "train": args.train,
        "val": args.val,
        "test": args.test,
        "num_iq_samples": args.num_iq_samples,
        "sample_rate": args.sample_rate,
        "preset": args.preset,
        "num_signals_min": args.num_signals_min,
        "num_signals_max": args.num_signals_max,
        "impairment_level": args.impairment_level,
        "fft_size": args.fft_size,
        "stft_fft": args.stft_fft,
        "stft_hop": args.stft_hop,
        "snr_db_min": args.snr_db_min,
        "snr_db_max": args.snr_db_max,
        "snr_buckets": [[float(lo), float(hi)] for lo, hi in args.snr_buckets],
        "duration_min_frac": args.duration_min_frac,
        "duration_max_frac": args.duration_max_frac,
        "bandwidth_min_frac": args.bandwidth_min_frac,
        "bandwidth_max_frac": args.bandwidth_max_frac,
        "center_freq_min_frac": args.center_freq_min_frac,
        "center_freq_max_frac": args.center_freq_max_frac,
        "cochannel_overlap_probability": args.cochannel_overlap_probability,
        "fast_snr_update": bool(args.fast_snr_update),
        "snr_update_method": "time_domain_power_scaling" if args.fast_snr_update else "torchsig_spectrogram_refinement",
        "categories": categories,
        "provenance": generation_provenance(args),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[prepare-iq-stratified] wrote {out_root}", flush=True)


if __name__ == "__main__":
    main()
