#!/usr/bin/env python
"""Build the manifest's release configs without data access or training."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from pathlib import Path


def repo_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "tools" / "train.py").exists() and (candidate / "csrr").is_dir():
            return candidate
    raise RuntimeError("CSRR repository root not found")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build every manifest build_config; never load data or train.",
    )
    args = parser.parse_args()
    if not args.build_only:
        parser.error("--build-only is required; this release gate never trains")

    root = repo_root()
    paper_dir = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(paper_dir) not in sys.path:
        sys.path.insert(0, str(paper_dir))

    from run_mmdet_smoke import (
        maybe_stub_mmcv_ext,
        patch_focal_loss_for_mmcv_lite,
        patch_msda_for_mmcv_lite,
        patch_roi_align_for_mmcv_lite,
    )

    used_stub = maybe_stub_mmcv_ext()
    patch_results = {
        "focal_loss": patch_focal_loss_for_mmcv_lite(),
        "roi_align": patch_roi_align_for_mmcv_lite(),
        "multi_scale_deformable_attention": patch_msda_for_mmcv_lite(),
    }

    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.utils import import_modules_from_strings
    from mmdet.registry import MODELS

    manifest = json.loads((paper_dir / "paper_manifest.json").read_text(encoding="utf-8"))
    failures = []
    print(f"[release] mmcv-lite stub={used_stub} patches={patch_results}", flush=True)
    for relative in manifest["build_configs"]:
        config_path = root / relative
        try:
            cfg = Config.fromfile(str(config_path))
            custom_imports = cfg.get("custom_imports")
            if custom_imports:
                import_modules_from_strings(**custom_imports)
            init_default_scope(cfg.get("default_scope", "mmdet"))
            model = MODELS.build(cfg.model)
            parameters = sum(parameter.numel() for parameter in model.parameters())
            print(f"[PASS] {relative} params={parameters:,}", flush=True)
            del model
            gc.collect()
        except Exception as exc:  # noqa: BLE001
            failures.append(relative)
            print(f"[FAIL] {relative}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()

    print(
        f"[release] RESULT: {'FAILED' if failures else 'OK'} "
        f"({len(manifest['build_configs']) - len(failures)}/"
        f"{len(manifest['build_configs'])} built)",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
