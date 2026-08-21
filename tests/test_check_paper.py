import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = next(
    candidate for candidate in Path(__file__).resolve().parents
    if (candidate / "tools" / "misc" / "check_paper.py").exists()
)
SPEC = importlib.util.spec_from_file_location(
    "check_paper", ROOT / "tools" / "misc" / "check_paper.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def write(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_repo(tmp_path, level="statistical"):
    root = tmp_path / "repo"
    paper = root / "configs" / "demo"
    write(root / "tools" / "train.py")
    write(root / "csrr" / "__init__.py")
    write(root / "requirements" / "runtime.txt", "mmengine==0.10.7\n")
    write(root / "requirements" / "demo.txt", "numpy==2.2.6\n")
    row = "| [Demo](configs/demo) | [Demo Paper](configs/demo) |\n"
    write(root / "README.md", row)
    write(root / "README_zh-CN.md", row)
    write(
        paper / "README.md",
        "English | [简体中文](README_zh-CN.md)\n\n"
        f"Reproduction level: `{level}`\n",
    )
    write(
        paper / "README_zh-CN.md",
        "[English](README.md) | 简体中文\n\n"
        f"复现等级：`{level}`\n",
    )
    config = "configs/demo/demo_iq-data.py"
    write(root / config, "# Paper: \"Demo\", Venue (under review).\nmodel = dict(type='Demo')\n")
    write(paper / "helper.py", "print('helper')\n")
    manifest = {
        "schema_version": 1,
        "name": "demo",
        "paper": {
            "title": "Demo Paper",
            "venue": "Demo Venue",
            "status": "under review",
        },
        "official_configs": [config],
        "build_configs": [config],
        "runtime_check": ["{python}", "-c", "raise SystemExit(0)"],
        "requirements": "requirements/demo.txt",
        "reproduction_level": level,
        "known_limitations": ["different realization"] if level != "exact" else [],
        "external_framework_exceptions": [],
        "declared_core_changes": [],
    }
    write(paper / "paper_manifest.json", json.dumps(manifest, indent=2))
    return root, manifest


@pytest.mark.parametrize(
    "value",
    [
        'ROOT = "' + "/" + 'data/private/run"',
        'ROOT = "' + "C" + r':\private\run"',
        'ROOT = "' + "\\" + r'\server\share\run"',
        "host = '" + "10" + ".161.4.39'",
        "host = 'user@" + "192" + ".168.1.20'",
        "root = Path(__file__).resolve()." + "parents" + "[4]",
    ],
)
def test_machine_references_are_rejected(value):
    assert CHECK.machine_reference_reasons(value)


@pytest.mark.parametrize(
    "value",
    [
        "https://github.com/org/repo/data/file",
        "read from <repo>/data/file",
        "write to <work-dir>\\results",
        r'ax.text(0, 0, "recognition $\\sim\\!0.5$")',
    ],
)
def test_public_urls_and_placeholders_are_not_paths(value):
    assert CHECK.machine_reference_reasons(value) == []


@pytest.mark.parametrize(
    "value",
    [
        'ROOT = "' + "/" + 'tmp/citybuster/run"',
        'ROOT = "' + "/" + 'opt/data/run"',
        'ROOT = "' + "/" + 'var/lib/run"',
        'ROOT = "' + "/" + 'srv/share/run"',
    ],
)
def test_extra_machine_roots_are_rejected(value):
    assert CHECK.machine_reference_reasons(value)


def test_usr_bin_env_shebang_is_not_flagged():
    # /usr is deliberately excluded so the standard shebang is not a finding.
    assert CHECK.machine_reference_reasons("#!" + "/usr/bin/env python") == []


def test_csv_ledgers_are_scanned_for_machine_paths(tmp_path):
    # Result ledgers ship as CSV; a server path in a comment row must not slip through.
    ledger = tmp_path / "results.csv"
    ledger.write_text("# measured on user@" + "10" + ".0.0.9:" + "/" + "data/run\ncell,value\na,1\n",
                      encoding="utf-8")
    assert CHECK.scan_machine_references([ledger])


def test_premerge_allows_deleting_a_side_directory(tmp_path):
    # Removing a legacy docs/<name>/ file is the convention being applied, not violated:
    # only added or modified files may land in a forbidden side path.
    root, manifest = make_repo(tmp_path)
    write(root / "docs" / "demo" / "OLD.md", "legacy notes\n")
    git(root, "init")
    git(root, "config", "user.name", "ChangShuo")
    git(root, "config", "user.email", "changshuo@bupt.edu.cn")
    git(root, "add", ".")
    git(root, "commit", "-m", "Base")
    git(root, "branch", "-M", "main")
    git(root, "switch", "-c", "paper/demo")
    git(root, "rm", "-q", "docs/demo/OLD.md")
    git(root, "commit", "-m", "Drop legacy side directory")
    report = CHECK.validate_premerge(root, "demo", "main", manifest)
    row = next(r for r in report.rows if r[0].startswith("diff contains no manuscripts"))
    assert row[2], row


def test_fake_allow_marker_does_not_bypass_detection():
    value = 'ROOT = "' + "/" + 'data/private/run"  # check-paper: allow-example'
    assert CHECK.machine_reference_reasons(value)


def test_checker_does_not_flag_its_own_detection_patterns():
    checker_path = ROOT / "tools" / "misc" / "check_paper.py"
    assert CHECK.scan_machine_references([checker_path]) == []


def test_path_boundary_does_not_accept_same_prefix(tmp_path):
    paper = tmp_path / "configs" / "name"
    evil = tmp_path / "configs" / "name_evil" / "config.py"
    paper.mkdir(parents=True)
    evil.parent.mkdir(parents=True)
    evil.touch()
    assert not CHECK.is_within(evil, paper)


def test_manifest_missing_fields_fails(tmp_path):
    root, manifest = make_repo(tmp_path)
    manifest.pop("runtime_check")
    write(root / "configs" / "demo" / "paper_manifest.json", json.dumps(manifest))
    report, _ = CHECK.validate_static(root, "demo", run_import_probe=False)
    assert report.failed()
    assert any("required fields" in row[0] and not row[2] for row in report.rows)


def test_manifest_escape_fails(tmp_path):
    root, manifest = make_repo(tmp_path)
    manifest["official_configs"] = ["../outside.py"]
    write(root / "configs" / "demo" / "paper_manifest.json", json.dumps(manifest))
    report, _ = CHECK.validate_static(root, "demo", run_import_probe=False)
    assert report.failed()
    assert any("official config paths" in row[0] and not row[2] for row in report.rows)


def test_missing_official_config_header_fails(tmp_path):
    root, _ = make_repo(tmp_path)
    write(root / "configs" / "demo" / "demo_iq-data.py", "model = dict(type='Demo')\n")
    report, _ = CHECK.validate_static(root, "demo", run_import_probe=False)
    assert report.failed()
    assert any("# Paper:" in row[0] and not row[2] for row in report.rows)


def test_unlisted_script_is_not_treated_as_config(tmp_path):
    root, _ = make_repo(tmp_path)
    report, _ = CHECK.validate_static(root, "demo", run_import_probe=True)
    assert not report.failed(), report.rows


def test_statistical_requires_limitations(tmp_path):
    root, manifest = make_repo(tmp_path)
    manifest["known_limitations"] = []
    write(root / "configs" / "demo" / "paper_manifest.json", json.dumps(manifest))
    report, _ = CHECK.validate_static(root, "demo", run_import_probe=False)
    assert report.failed()
    assert any("limitations" in row[0] and not row[2] for row in report.rows)


def test_changed_core_must_be_declared():
    changed = ["configs/demo/a.py", "csrr/visualization/visualizer.py"]
    assert CHECK.undeclared_core_changes(changed, {"declared_core_changes": []}) == [
        "csrr/visualization/visualizer.py"
    ]


@pytest.mark.parametrize("package", CHECK.MM_FAMILY)
def test_core_probe_blocks_every_mm_family(tmp_path, package):
    root = tmp_path / package
    write(root / "csrr" / "__init__.py", f"import {package}\n")
    ok, detail = CHECK._blocked_import_probe(root)
    assert not ok
    assert "blocked" in detail


def test_branch_policy():
    assert CHECK.branch_is_valid("paper/demo", "demo")
    assert CHECK.branch_is_valid("paper/demo-taxonomy", "demo")
    assert not CHECK.branch_is_valid("demo-taxonomy", "demo")
    assert not CHECK.branch_is_valid("paper/demo/extra", "demo")


def test_commit_metadata_policy():
    good = {
        "author_name": "ChangShuo",
        "author_email": "changshuo@bupt.edu.cn",
        "committer_name": "ChangShuo",
        "committer_email": "changshuo@bupt.edu.cn",
        "message": "One line",
    }
    assert CHECK.validate_commit_record(good) == []
    bad = dict(good)
    bad.update(
        author_name="Assistant",
        message="Subject\n\nbody\nCo-authored-by: Bot <bot@example.com>",
    )
    problems = CHECK.validate_commit_record(bad)
    assert "wrong author" in problems
    assert "commit message/body is not one line" in problems
    assert "Co-authored-by present" in problems


def test_runtime_uses_argv_without_shell(tmp_path):
    root, manifest = make_repo(tmp_path)
    observed = {}

    def runner(argv, cwd, shell):
        observed.update(argv=argv, cwd=cwd, shell=shell)
        return SimpleNamespace(returncode=0)

    report = CHECK.validate_runtime(root, manifest, runner=runner)
    assert not report.failed()
    assert observed["shell"] is False
    assert observed["argv"][0] == sys.executable


def test_runtime_propagates_failure(tmp_path):
    root, manifest = make_repo(tmp_path)

    def runner(argv, cwd, shell):
        return SimpleNamespace(returncode=7)

    report = CHECK.validate_runtime(root, manifest, runner=runner)
    assert report.failed()


def git(root, *args, env=None):
    return subprocess.run(
        ["git", *args],
        cwd=str(root),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_premerge_rejects_trailing_whitespace(tmp_path):
    root, manifest = make_repo(tmp_path)
    git(root, "init")
    git(root, "config", "user.name", "ChangShuo")
    git(root, "config", "user.email", "changshuo@bupt.edu.cn")
    git(root, "add", ".")
    git(root, "commit", "-m", "Base")
    git(root, "branch", "-M", "main")
    git(root, "switch", "-c", "paper/demo")
    with (root / "README.md").open("a", encoding="utf-8") as stream:
        stream.write("trailing space   \n")
    git(root, "add", "README.md")
    git(root, "commit", "-m", "Paper update")
    report = CHECK.validate_premerge(root, "demo", "main", manifest)
    assert report.failed()
    assert any("diff --check" in row[0] and not row[2] for row in report.rows)


def test_premerge_rejects_dirty_worktree(tmp_path):
    root, manifest = make_repo(tmp_path)
    git(root, "init")
    git(root, "config", "user.name", "ChangShuo")
    git(root, "config", "user.email", "changshuo@bupt.edu.cn")
    git(root, "add", ".")
    git(root, "commit", "-m", "Base")
    git(root, "branch", "-M", "main")
    git(root, "switch", "-c", "paper/demo")
    with (root / "README.md").open("a", encoding="utf-8") as stream:
        stream.write("uncommitted\n")
    report = CHECK.validate_premerge(root, "demo", "main", manifest)
    assert report.failed()
    assert any("worktree and index" in row[0] and not row[2] for row in report.rows)
