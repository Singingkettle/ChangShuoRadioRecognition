"""Regression tests for the mmengine-only core (de-mmcv).

The csrr core was changed to depend on mmengine only: mmcv/mmdet must not be
unconditional top-level imports of the core (they are isolated to individual
papers). These tests guard the de-mmcv edits to ``csrr/__init__.py`` and
``csrr/utils/collect_env.py`` against reintroduction.
"""
import ast
from pathlib import Path

import pytest

ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "csrr" / "__init__.py").exists()
)


def _toplevel_import_roots(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots = set()
    for node in tree.body:  # module top level only
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("relpath", [
    "csrr/__init__.py",
    "csrr/utils/collect_env.py",
])
def test_core_file_has_no_unconditional_mm_import(relpath):
    roots = _toplevel_import_roots(ROOT / relpath)
    assert "mmcv" not in roots, f"{relpath} imports mmcv at module top level"
    assert "mmdet" not in roots, f"{relpath} imports mmdet at module top level"


def test_collect_env_runs_and_mmcv_is_optional():
    pytest.importorskip("mmengine")
    from csrr.utils.collect_env import collect_env

    info = collect_env()
    assert isinstance(info, dict) and info
