# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


CANONICAL_CLASS_NAMES = [
    "1024qam",
    "128qam_cross",
    "16ask",
    "16fsk",
    "16gfsk",
    "16gmsk",
    "16msk",
    "16psk",
    "16qam",
    "2fsk",
    "2gfsk",
    "2gmsk",
    "2msk",
    "256qam",
    "32ask",
    "32psk",
    "32qam",
    "32qam_cross",
    "4ask",
    "4fsk",
    "4gfsk",
    "4gmsk",
    "4msk",
    "512qam_cross",
    "64ask",
    "64psk",
    "64qam",
    "8ask",
    "8fsk",
    "8gfsk",
    "8gmsk",
    "8msk",
    "8psk",
    "am-dsb",
    "am-dsb-sc",
    "am-lsb",
    "am-usb",
    "bpsk",
    "chirpss",
    "fm",
    "lfm-data",
    "lfm-radar",
    "ofdm-1024",
    "ofdm-1200",
    "ofdm-128",
    "ofdm-180",
    "ofdm-2048",
    "ofdm-256",
    "ofdm-300",
    "ofdm-512",
    "ofdm-600",
    "ofdm-64",
    "ofdm-72",
    "ofdm-900",
    "ook",
    "qpsk",
    "tone",
]
CANONICAL_CATEGORY_IDS = {name: idx for idx, name in enumerate(CANONICAL_CLASS_NAMES)}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    return str(value)


def metadata_to_dict(metadata: Any) -> dict[str, Any]:
    if isinstance(metadata, dict):
        return jsonable(dict(metadata))
    if hasattr(metadata, "to_dict"):
        raw = metadata.to_dict()
    else:
        raw = {
            k: v
            for k, v in getattr(metadata, "__dict__", {}).items()
            if not k.startswith("_") and k not in {"dataset_metadata", "applied_transforms"}
        }

    for attr in (
        "start",
        "stop",
        "duration",
        "lower_freq",
        "upper_freq",
        "_lower_frequency",
        "_upper_frequency",
        "sample_rate",
        "center_freq",
        "bandwidth",
        "class_name",
        "class_index",
        "snr_db",
        "start_in_samples",
        "duration_in_samples",
        "stop_in_samples",
    ):
        if attr not in raw and hasattr(metadata, attr):
            try:
                raw[attr] = getattr(metadata, attr)
            except Exception:
                pass
    return jsonable(raw)


def flatten_transforms(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def instance_frequency_bounds(instance: dict[str, Any]) -> tuple[float, float] | None:
    """Return frequency bounds in Hz for a TorchSig instance.

    TorchSig metadata often exposes private `_lower_frequency` and
    `_upper_frequency` fields, but these fields can disagree with the public
    `center_freq`/`bandwidth` pair for negative center frequencies. The public
    center/bandwidth pair is the stable source for time-frequency boxes.
    """

    center = instance.get("center_freq")
    bandwidth = instance.get("bandwidth")
    if center is not None and bandwidth is not None:
        center_f = float(center)
        bandwidth_f = abs(float(bandwidth))
        if math.isfinite(center_f) and math.isfinite(bandwidth_f) and bandwidth_f > 0.0:
            return center_f - 0.5 * bandwidth_f, center_f + 0.5 * bandwidth_f

    lower = instance.get("lower_freq")
    upper = instance.get("upper_freq")
    if lower is None:
        lower = instance.get("_lower_frequency")
    if upper is None:
        upper = instance.get("_upper_frequency")
    if lower is None or upper is None:
        return None
    lower_f = float(lower)
    upper_f = float(upper)
    if not math.isfinite(lower_f) or not math.isfinite(upper_f):
        return None
    return min(lower_f, upper_f), max(lower_f, upper_f)


def create_torchsig_dataset(args: argparse.Namespace, seed: int):
    from torchsig.datasets.datasets import TorchSigIterableDataset

    # The dict form is required: everything below (geometry, SNR range, co-channel
    # probability) is applied by updating it. The old code fell back to a DatasetMetadata
    # object on ANY exception, which skipped that whole block and generated with TorchSig's
    # stock defaults -- silently ignoring every argument this script was given.
    try:
        from torchsig.utils.defaults import TorchSigDefaults

        metadata = dict(TorchSigDefaults().default_dataset_metadata)
    except ImportError as exc:
        raise RuntimeError(
            "torchsig.utils.defaults.TorchSigDefaults is unavailable "
            f"({exc}). This script pins torchsig==2.1.1; a different version changes the "
            "dataset metadata contract and would silently generate a different benchmark."
        ) from exc
    if not isinstance(metadata, dict):
        raise RuntimeError(
            f"Expected TorchSigDefaults().default_dataset_metadata to be a dict, got "
            f"{type(metadata).__name__}. Refusing to generate with unapplied arguments."
        )

    if isinstance(metadata, dict):
        min_duration = max(16, int(args.duration_min_frac * args.num_iq_samples))
        max_duration = max(min_duration + 1, int(args.duration_max_frac * args.num_iq_samples))
        bandwidth_min = args.bandwidth_min_frac * args.sample_rate
        bandwidth_max = args.bandwidth_max_frac * args.sample_rate
        metadata.update(
            dict(
                num_iq_samples_dataset=args.num_iq_samples,
                fft_size=args.fft_size,
                num_signals_min=args.num_signals_min,
                num_signals_max=args.num_signals_max,
                num_samples=None,
                sample_rate=args.sample_rate,
                frequency_min=-0.5 * args.sample_rate,
                frequency_max=0.5 * args.sample_rate,
                signal_center_freq_min=args.center_freq_min_frac * args.sample_rate,
                signal_center_freq_max=args.center_freq_max_frac * args.sample_rate,
                bandwidth_min=bandwidth_min,
                bandwidth_max=bandwidth_max,
                signal_duration_in_samples_min=min_duration,
                signal_duration_in_samples_max=max_duration,
                snr_db_min=args.snr_db_min,
                snr_db_max=args.snr_db_max,
                cochannel_overlap_probability=args.cochannel_overlap_probability,
            )
        )
        metadata.setdefault("fft_stride", args.stft_hop)
        if args.noise_power_db is not None:
            metadata["noise_power_db"] = args.noise_power_db
        else:
            metadata.setdefault("noise_power_db", -60.0)

    transforms: list[Any] = []
    component_transforms: list[Any] = []
    try:
        from torchsig.transforms.impairments import Impairments

        impairments = Impairments(level=args.impairment_level)
        transforms.extend(flatten_transforms(getattr(impairments, "dataset_transforms", None)))
        component_transforms.extend(flatten_transforms(getattr(impairments, "signal_transforms", None)))
    except Exception as exc:
        raise RuntimeError(
            f"Could not construct TorchSig impairments at level {args.impairment_level}: {exc}. "
            "Generating without impairments would produce a different benchmark; refusing to "
            "continue. This script pins torchsig==2.1.1."
        ) from exc

    # Only seeded signatures are accepted. The previous version also tried unseeded
    # variants and a bare TorchSigIterableDataset(metadata) -- any TypeError walked
    # silently down to a dataset with no seed and, at the end, no transforms either,
    # which makes generation irreproducible without saying so.
    ctor_attempts = [
        dict(metadata=metadata, transforms=transforms, component_transforms=component_transforms, target_labels=None, seed=seed),
        dict(dataset_metadata=metadata, transforms=transforms, component_transforms=component_transforms, target_labels=None, seed=seed),
    ]
    last_error: Exception | None = None
    for kwargs in ctor_attempts:
        try:
            return TorchSigIterableDataset(**kwargs)
        except TypeError as exc:
            last_error = exc
    raise RuntimeError(
        "Could not construct a SEEDED TorchSigIterableDataset. Generation must be "
        f"reproducible, so no unseeded fallback is attempted. Last error: {last_error}. "
        "Check that torchsig==2.1.1 is installed (its constructor accepts `seed`)."
    )


def sample_from_dataset(dataset: Any) -> tuple[np.ndarray, list[dict[str, Any]]]:
    sample = next(iter(dataset))
    if isinstance(sample, tuple):
        data = np.asarray(sample[0])
        targets = sample[1]
        if targets and isinstance(targets[0], dict):
            return data, [jsonable(t) for t in targets]
        raise RuntimeError(
            "TorchSig returned tuple targets rather than Signal metadata. "
            "Use target_labels=None so full metadata is available."
        )

    data = np.asarray(sample.data)
    component_signals = getattr(sample, "component_signals", None) or []
    if component_signals:
        metadatas = [getattr(component, "metadata", {}) for component in component_signals]
    elif hasattr(sample, "get_full_metadata"):
        metadatas = sample.get_full_metadata()
    else:
        metadatas = []
    return data, [metadata_to_dict(m) for m in metadatas]


def sample_with_retries(
    dataset: Any,
    *,
    split: str,
    sample_idx: int,
    max_retries: int,
) -> tuple[np.ndarray, list[dict[str, Any]], int]:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            iq, instances = sample_from_dataset(dataset)
            if not np.all(np.isfinite(iq)):
                raise ValueError("generated IQ contains NaN or Inf")
            return iq, instances, attempt
        except Exception as exc:  # TorchSig can reject rare random impairment draws.
            last_error = exc
            if attempt < max_retries:
                print(
                    f"[prepare] warning: retrying {split} sample {sample_idx} "
                    f"after TorchSig error ({attempt + 1}/{max_retries}): {exc}"
                )
    raise RuntimeError(
        f"Could not generate {split} sample {sample_idx} after {max_retries} retries: {last_error}"
    )


def compute_stft_image(iq: np.ndarray, n_fft: int, hop: int, image_size: int) -> Image.Image:
    iq = np.asarray(iq, dtype=np.complex64)
    if iq.size < n_fft:
        iq = np.pad(iq, (0, n_fft - iq.size))
    starts = np.arange(0, max(1, iq.size - n_fft + 1), hop)
    if starts.size == 0:
        starts = np.array([0])
    window = np.hanning(n_fft).astype(np.float32)
    frames = np.stack([iq[s : s + n_fft] * window for s in starts], axis=0)
    spec = np.fft.fftshift(np.fft.fft(frames, n=n_fft, axis=1), axes=1)
    power = 20.0 * np.log10(np.abs(spec) + 1e-6)
    lo, hi = np.percentile(power, [2.0, 98.0])
    norm = np.clip((power - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    img = (255.0 * norm.T[::-1, :]).astype(np.uint8)
    return Image.fromarray(img, mode="L").resize((image_size, image_size), Image.Resampling.BILINEAR).convert("RGB")


def instance_box(
    instance: dict[str, Any],
    num_iq_samples: int,
    sample_rate: float,
    image_size: int,
) -> list[float] | None:
    start = instance.get("start")
    stop = instance.get("stop")
    if start is None:
        start_samples = float(instance.get("start_in_samples", 0))
        start = start_samples / max(num_iq_samples, 1)
    if stop is None:
        if instance.get("stop_in_samples") is not None:
            stop_samples = float(instance["stop_in_samples"])
        else:
            stop_samples = float(instance.get("start_in_samples", 0)) + float(instance.get("duration_in_samples", 0))
        stop = stop_samples / max(num_iq_samples, 1)

    bounds = instance_frequency_bounds(instance)
    if bounds is None:
        return None
    lower, upper = bounds

    start = float(np.clip(start, 0.0, 1.0))
    stop = float(np.clip(stop, 0.0, 1.0))
    lower_norm = float(np.clip(float(lower) / sample_rate + 0.5, 0.0, 1.0))
    upper_norm = float(np.clip(float(upper) / sample_rate + 0.5, 0.0, 1.0))

    x0 = min(start, stop) * image_size
    x1 = max(start, stop) * image_size
    y0 = (1.0 - max(lower_norm, upper_norm)) * image_size
    y1 = (1.0 - min(lower_norm, upper_norm)) * image_size
    w = x1 - x0
    h = y1 - y0
    if w <= 1.0 or h <= 1.0 or not all(math.isfinite(v) for v in [x0, y0, w, h]):
        return None
    return [round(x0, 3), round(y0, 3), round(w, 3), round(h, 3)]


def class_key(instance: dict[str, Any]) -> tuple[int, str]:
    idx = instance.get("class_index")
    name = instance.get("class_name")
    if idx is None:
        idx = -1
    if not name:
        name = f"class_{idx}"
    return int(idx), str(name)


def category_id_for_key(
    key: tuple[int, str],
    category_map: dict[tuple[int, str], int],
    next_category_id: int,
) -> tuple[int, int]:
    """Return a stable category id for TorchSig classes.

    Earlier smoke datasets assigned ids in encounter order, which made
    checkpoints unsafe to transfer across generated splits. Known TorchSig
    classes now use a fixed vocabulary; unknown classes are appended
    deterministically within a generation run.
    """

    canonical_id = CANONICAL_CATEGORY_IDS.get(key[1].lower())
    if canonical_id is not None:
        category_map[key] = canonical_id
        return canonical_id, next_category_id
    if key not in category_map:
        category_map[key] = next_category_id
        next_category_id += 1
    return category_map[key], next_category_id


def categories_from_map(category_map: dict[tuple[int, str], int]) -> list[dict[str, Any]]:
    categories = [
        {"id": cat_id, "name": name, "torchsig_class_index": -1}
        for name, cat_id in CANONICAL_CATEGORY_IDS.items()
    ]
    canonical_ids = set(CANONICAL_CATEGORY_IDS.values())
    unknown_categories = [
        {"id": cat_id, "name": name, "torchsig_class_index": idx}
        for (idx, name), cat_id in category_map.items()
        if cat_id not in canonical_ids
    ]
    categories.extend(sorted(unknown_categories, key=lambda item: item["id"]))
    return sorted(categories, key=lambda item: item["id"])


def render_preview(image_path: Path, annotations: list[dict[str, Any]], out_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        draw.rectangle((x, y, x + w, y + h), outline=(255, 80, 40), width=3)
    image.save(out_path)


def write_split(
    split: str,
    count: int,
    args: argparse.Namespace,
    out_root: Path,
    category_map: dict[tuple[int, str], int],
    next_category_id: int,
) -> int:
    raw_dir = out_root / "raw" / split
    img_dir = out_root / "coco" / split / "images"
    ann_dir = out_root / "coco" / "annotations"
    meta_dir = out_root / "metadata"
    for path in (raw_dir, img_dir, ann_dir, meta_dir):
        path.mkdir(parents=True, exist_ok=True)

    dataset = create_torchsig_dataset(args, seed=args.seed + {"train": 0, "val": 1000, "test": 2000}[split])

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 1
    metadata_lines: list[str] = []

    retry_count = 0
    for sample_idx in range(count):
        iq, instances, retries = sample_with_retries(
            dataset,
            split=split,
            sample_idx=sample_idx,
            max_retries=args.max_sample_retries,
        )
        retry_count += retries
        sample_id = f"{split}_{sample_idx:06d}"
        npz_path = raw_dir / f"{sample_id}.npz"
        np.savez_compressed(npz_path, iq=iq.astype(np.complex64))

        image = compute_stft_image(iq, args.stft_fft, args.stft_hop, args.image_size)
        image_name = f"{sample_id}.png"
        image_path = img_dir / image_name
        image.save(image_path)

        image_id = len(images) + 1
        images.append(
            {
                "id": image_id,
                "file_name": image_name,
                "width": args.image_size,
                "height": args.image_size,
            }
        )

        serial_instances: list[dict[str, Any]] = []
        sample_rate = float(instances[0].get("sample_rate", args.sample_rate)) if instances else args.sample_rate
        for inst in instances:
            key = class_key(inst)
            cat_id, next_category_id = category_id_for_key(key, category_map, next_category_id)
            bbox = instance_box(inst, len(iq), sample_rate, args.image_size)
            inst = dict(inst)
            inst["category_id"] = cat_id
            serial_instances.append(inst)
            if bbox is None:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": bbox,
                    "area": round(bbox[2] * bbox[3], 3),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        metadata_lines.append(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "raw_path": str(npz_path.as_posix()),
                    "image_path": str(image_path.as_posix()),
                    "num_iq_samples": int(len(iq)),
                    "sample_rate": sample_rate,
                    "instances": serial_instances,
                },
                ensure_ascii=False,
            )
        )
        if (sample_idx + 1) % 100 == 0 or sample_idx + 1 == count:
            print(f"[prepare] {split}: generated {sample_idx + 1}/{count} samples")

    categories = categories_from_map(category_map)
    coco = {"images": images, "annotations": annotations, "categories": categories}
    (ann_dir / f"instances_{split}.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")
    (meta_dir / f"{split}.jsonl").write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")

    if split == "train" and images:
        first_image = img_dir / images[0]["file_name"]
        first_anns = [ann for ann in annotations if ann["image_id"] == 1]
        render_preview(first_image, first_anns, out_root / "preview_train_000000.png")

    print(
        f"[prepare] {split}: {len(images)} images, {len(annotations)} boxes, "
        f"{len(categories)} categories, {retry_count} retries"
    )
    return next_category_id


def rewrite_global_categories(out_root: Path, category_map: dict[tuple[int, str], int]) -> list[dict[str, Any]]:
    categories = categories_from_map(category_map)
    ann_dir = out_root / "coco" / "annotations"
    for split in ("train", "val", "test"):
        ann_path = ann_dir / f"instances_{split}.json"
        coco = json.loads(ann_path.read_text(encoding="utf-8"))
        coco["categories"] = categories
        ann_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    return categories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="data/torchsig_mini")
    parser.add_argument(
        "--preset",
        choices=["custom", "wbsig53-paper", "wbsig53-clean-like", "torchsig-wideband-default"],
        default="custom",
        help="Apply a reproducible dataset preset before generation.",
    )
    parser.add_argument("--train", type=int, default=16)
    parser.add_argument("--val", type=int, default=4)
    parser.add_argument("--test", type=int, default=4)
    parser.add_argument("--num-iq-samples", type=int, default=4096)
    parser.add_argument("--sample-rate", type=float, default=1_000_000.0)
    parser.add_argument("--num-signals-min", type=int, default=1)
    parser.add_argument("--num-signals-max", type=int, default=3)
    parser.add_argument("--impairment-level", type=int, default=2)
    parser.add_argument("--fft-size", type=int, default=256)
    parser.add_argument("--stft-fft", type=int, default=256)
    parser.add_argument("--stft-hop", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--duration-min-frac", type=float, default=0.12)
    parser.add_argument("--duration-max-frac", type=float, default=0.60)
    parser.add_argument("--bandwidth-min-frac", type=float, default=0.04)
    parser.add_argument("--bandwidth-max-frac", type=float, default=0.25)
    parser.add_argument("--center-freq-min-frac", type=float, default=-0.35)
    parser.add_argument("--center-freq-max-frac", type=float, default=0.35)
    parser.add_argument("--snr-db-min", type=float, default=0.0)
    parser.add_argument("--snr-db-max", type=float, default=50.0)
    parser.add_argument("--cochannel-overlap-probability", type=float, default=0.2)
    parser.add_argument("--noise-power-db", type=float, default=None)
    parser.add_argument("--max-sample-retries", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def apply_dataset_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.preset not in {"wbsig53-paper", "wbsig53-clean-like", "torchsig-wideband-default"}:
        return args

    args.num_iq_samples = 262_144
    args.sample_rate = 10_000_000.0
    args.fft_size = 512
    args.stft_fft = 512
    args.stft_hop = 512
    args.image_size = 512

    if args.preset == "torchsig-wideband-default":
        # Matches TorchSig 2.1.1 source default_configs/wideband_*_all.yaml.
        args.num_signals_min = 3
        args.num_signals_max = 5
        args.impairment_level = 0
        args.snr_db_min = 0.0
        args.snr_db_max = 50.0
        args.duration_min_frac = 16_384 / 262_144
        args.duration_max_frac = 32_768 / 262_144
        args.bandwidth_min_frac = 62_500 / 10_000_000
        args.bandwidth_max_frac = 1_000_000 / 10_000_000
        args.center_freq_min_frac = -0.40
        args.center_freq_max_frac = 0.40
        args.cochannel_overlap_probability = 0.1
        return args

    args.num_signals_min = 1
    args.num_signals_max = 6
    if args.preset == "wbsig53-clean-like":
        args.impairment_level = 0
        args.snr_db_min = 20.0
        args.snr_db_max = 40.0
    else:
        args.impairment_level = 2
        args.snr_db_min = 0.0
        args.snr_db_max = 30.0
    args.duration_min_frac = 0.05
    args.duration_max_frac = 1.0
    args.bandwidth_min_frac = 0.0125
    # WBSig53 reports normalized occupied bandwidths up to 0.7, but the
    # TorchSig 2.1.1 generator used here rejects bandwidths >= sample_rate/2.
    # Keep the preset legally generatable and record this as a TorchSig-v2
    # compatible approximation rather than an exact WBSig53 dataset rebuild.
    args.bandwidth_max_frac = 0.49
    args.center_freq_min_frac = -0.40
    args.center_freq_max_frac = 0.40
    args.cochannel_overlap_probability = 0.0
    return args


def main() -> None:
    args = parse_args()
    args = apply_dataset_preset(args)
    out_root = repo_root() / args.out_root
    if out_root.exists() and args.force:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    category_map: dict[tuple[int, str], int] = {}
    next_category_id = len(CANONICAL_CLASS_NAMES)
    for split, count in (("train", args.train), ("val", args.val), ("test", args.test)):
        next_category_id = write_split(split, count, args, out_root, category_map, next_category_id)

    categories = rewrite_global_categories(out_root, category_map)
    summary = {
        "out_root": str(out_root),
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
        "duration_min_frac": args.duration_min_frac,
        "duration_max_frac": args.duration_max_frac,
        "bandwidth_min_frac": args.bandwidth_min_frac,
        "bandwidth_max_frac": args.bandwidth_max_frac,
        "center_freq_min_frac": args.center_freq_min_frac,
        "center_freq_max_frac": args.center_freq_max_frac,
        "cochannel_overlap_probability": args.cochannel_overlap_probability,
        "categories": categories,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[prepare] wrote {out_root}")


if __name__ == "__main__":
    main()
