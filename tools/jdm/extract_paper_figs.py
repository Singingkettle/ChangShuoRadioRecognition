#!/usr/bin/env python3
"""Provenance helper for paper figure digitization.

Rasterizes a local copy of arXiv:2405.00736 (not vendored in this repo) and
points at docs/csrd_jointdet/paper_figure_targets.md.
Does not auto-OCR radar charts; numeric tables are hand-digitized (±0.03–0.04).
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_TARGETS = _REPO / "docs/csrd_jointdet/paper_figure_targets.md"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf", type=Path, required=True,
        help="local path to 2405.00736.pdf (download from arXiv; not in git)")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("work_dirs/jdm/paper_pages"))
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    pdf = args.pdf.expanduser().resolve()
    if not pdf.is_file():
        raise SystemExit(f"Missing PDF: {pdf}")
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        ["pdftoppm", "-png", "-r", str(args.dpi), str(pdf), str(out / "page")]
    )
    print(f"Rasterized {pdf} -> {out}")
    print(f"Numeric targets: {_TARGETS}")


if __name__ == "__main__":
    main()
