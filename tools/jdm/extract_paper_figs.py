#!/usr/bin/env python3
"""Provenance helper for paper figure digitization.

Rasterizes arXiv:2405.00736 pages and points at docs/csrd_jointdet/paper_figure_targets.md.
Does not auto-OCR radar charts; numeric tables are hand-digitized (±0.03–0.04).
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_PDF = _REPO / "docs/csrd_jointdet/paper_assets/2405.00736.pdf"
_OUT = _REPO / "docs/csrd_jointdet/paper_assets/pages"
_TARGETS = _REPO / "docs/csrd_jointdet/paper_figure_targets.md"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    if not _PDF.is_file():
        raise SystemExit(f"Missing PDF: {_PDF}")
    _OUT.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        ["pdftoppm", "-png", "-r", str(args.dpi), str(_PDF), str(_OUT / "page")]
    )
    print(f"Rasterized {_PDF} -> {_OUT}")
    print(f"Numeric targets: {_TARGETS}")


if __name__ == "__main__":
    main()
