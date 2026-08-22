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
    write(paper / "release_check.py", "raise SystemExit(0)\n")
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
        "runtime_check": ["{python}", "configs/demo/release_check.py"],
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


def test_generated_latex_string_is_not_mistaken_for_unc():
    value = r'line = "\\footnotesize\\setlength{\\tabcolsep}{5pt}"'
    assert CHECK.machine_reference_reasons(value) == []


@pytest.mark.parametrize(
    "value",
    [
        chr(92) * 2 + "server" + chr(92) + "private" + chr(92) + "run",
        'path = "' + chr(92) * 4 + "server" + chr(92) * 2 + "private"
        + chr(92) * 2 + 'run"',
    ],
)
def test_literal_and_source_escaped_unc_are_rejected(value):
    assert CHECK.machine_reference_reasons(value)


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


def test_declared_external_framework_must_be_exactly_pinned(tmp_path):
    root, manifest = make_repo(tmp_path)
    manifest["external_framework_exceptions"] = [{
        "package": "mmdet", "scope": "configs/demo",
        "reason": "the reported experiment used this framework",
    }]
    write(root / "configs" / "demo" / "paper_manifest.json", json.dumps(manifest))
    report, _ = CHECK.validate_static(root, "demo", run_import_probe=False)
    assert report.failed()
    row = next(item for item in report.rows if "external frameworks are exactly pinned" in item[0])
    assert not row[2] and "mmdet" in row[3]

    write(root / "requirements" / "demo.txt", "mmdet==3.3.0\n")
    report, _ = CHECK.validate_static(root, "demo", run_import_probe=False)
    row = next(item for item in report.rows if "external frameworks are exactly pinned" in item[0])
    assert row[2]


def test_external_base_namespace_requires_manifest_declaration(tmp_path):
    root, manifest = make_repo(tmp_path)
    config = root / "configs" / "demo" / "demo_iq-data.py"
    write(config, "# Paper: Demo\n_base_ = 'mmdet::retinanet/example.py'\n")
    report, _ = CHECK.validate_static(root, "demo", run_import_probe=False)
    row = next(item for item in report.rows if item[0].startswith("_base_ stays"))
    assert not row[2]

    manifest["external_framework_exceptions"] = [{
        "package": "mmdet", "scope": "configs/demo",
        "reason": "same framework as the measured experiment",
    }]
    write(root / "requirements" / "demo.txt", "mmdet==3.3.0\n")
    write(root / "configs" / "demo" / "paper_manifest.json", json.dumps(manifest))
    report, _ = CHECK.validate_static(root, "demo", run_import_probe=False)
    row = next(item for item in report.rows if item[0].startswith("_base_ stays"))
    assert row[2]


def test_changed_core_must_be_declared():
    changed = ["configs/demo/a.py", "csrr/visualization/visualizer.py"]
    assert CHECK.undeclared_core_changes(changed, {"declared_core_changes": []}) == [
        "csrr/visualization/visualizer.py"
    ]


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




def git(root, *args, env=None, input=None):
    return subprocess.run(
        ["git", *args],
        cwd=str(root),
        env=env,
        input=input,
        check=True,
        capture_output=True,
        text=True,
    )


def init_paper_branch(root):
    git(root, "init")
    git(root, "config", "user.name", "ChangShuo")
    git(root, "config", "user.email", "changshuo@bupt.edu.cn")
    git(root, "add", ".")
    git(root, "commit", "-m", "Base")
    git(root, "branch", "-M", "main")
    git(root, "switch", "-c", "paper/demo")


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


# ---------------------------------------------------------------------------
# Round-3 adversarial probes turned into regression tests.
# ---------------------------------------------------------------------------


def test_wildcard_and_arbitrary_equality_are_not_exact_pins():
    pins = CHECK.pinned_requirement_names
    assert pins("mmdet==3.*\n") == set()
    assert pins("mmdet===3.3.0\n") == set()
    assert pins("mmdet==3.3.0,!=3.3.1\n") == set()
    assert pins("mmdet==3.3.0; python_version>='3.10'\n") == set()
    assert pins("mmdet==3.3.0  # the measured version\n") == {"mmdet"}
    assert pins("MMCV_Lite==2.1.0\n") == {"mmcv-lite"}
    assert pins("mmdet @ git+https://example.org/mmdetection@" + "a" * 40 + "\n") == {"mmdet"}


def test_requirement_includes_are_followed_inside_repo(tmp_path):
    root, manifest = make_repo(tmp_path)
    write(root / "requirements" / "demo.txt", "-r demo-extra.txt\nnumpy==2.2.6\n")
    write(root / "requirements" / "demo-extra.txt", "mmdet>=3.3\n")
    report, _ = CHECK.validate_static(root, "demo", run_import_probe=False)
    row = next(r for r in report.rows if r[0].startswith("paper requirements use exact"))
    assert not row[2] and "mmdet>=3.3" in row[3]

    write(root / "requirements" / "demo-extra.txt", "mmdet==3.3.0\n")
    manifest["external_framework_exceptions"] = [{
        "package": "mmdet", "scope": "configs/demo", "reason": "measured with it",
    }]
    write(root / "configs" / "demo" / "paper_manifest.json", json.dumps(manifest))
    report, _ = CHECK.validate_static(root, "demo", run_import_probe=False)
    assert next(r for r in report.rows if "exactly pinned" in r[0])[2]
    assert next(r for r in report.rows if r[0].startswith("paper requirements use exact"))[2]


def test_requirement_include_outside_repo_or_editable_fails(tmp_path):
    root, _ = make_repo(tmp_path)
    write(root / "requirements" / "demo.txt", "-r ../../outside.txt\n-e .\nnumpy==2.2.6\n")
    report, _ = CHECK.validate_static(root, "demo", run_import_probe=False)
    row = next(r for r in report.rows if r[0].startswith("paper requirements use exact"))
    assert not row[2] and "include" in row[3] and "editable" in row[3]


@pytest.mark.parametrize(
    "argv",
    [["bash", "-c", "echo hi"], ["{python}", "-c", "pass"], ["{python}", "-m", "pytest"],
     ["sh", "configs/demo/release_check.py"]],
)
def test_runtime_check_must_be_python_plus_repo_script(tmp_path, argv):
    root, manifest = make_repo(tmp_path)
    manifest["runtime_check"] = argv
    write(root / "configs" / "demo" / "paper_manifest.json", json.dumps(manifest))
    report, _ = CHECK.validate_static(root, "demo", run_import_probe=False)
    assert any(r[0].startswith("runtime_check is") and not r[2] for r in report.rows)
    with pytest.raises(ValueError):
        CHECK.runtime_argv(root, manifest)


def test_inline_co_authored_by_is_rejected():
    record = {
        "author_name": "ChangShuo", "author_email": "changshuo@bupt.edu.cn",
        "committer_name": "ChangShuo", "committer_email": "changshuo@bupt.edu.cn",
        "message": "Fix x Co-Authored-By: Bot <bot@example.com>",
    }
    assert "Co-authored-by present" in CHECK.validate_commit_record(record)


def test_control_characters_cannot_hide_a_trailer(tmp_path):
    root, manifest = make_repo(tmp_path)
    init_paper_branch(root)
    with (root / "README.md").open("a", encoding="utf-8") as stream:
        stream.write("more\n")
    git(root, "add", "README.md")
    git(root, "commit", "-q", "-F", "-", input="One line\x1eCo-authored-by: Bot <bot@example.com>")
    report = CHECK.validate_premerge(root, "demo", "main", manifest)
    assert next(r for r in report.rows if r[0] == "commit records are parseable")[2]
    row = next(r for r in report.rows if r[0].startswith("commit author/committer/message"))
    assert not row[2] and "Co-authored-by" in row[3]


def test_suffixless_and_notebook_files_are_scanned(tmp_path):
    docker = tmp_path / "Dockerfile"
    docker.write_text("WORKDIR " + "/" + "data/private\n", encoding="utf-8")
    notebook = tmp_path / "x.ipynb"
    notebook.write_text('{"cells": ["' + "C" + ':\\\\private"]}', encoding="utf-8")
    assert CHECK.scan_machine_references([docker, notebook])


def test_posix_style_unc_and_network_urls_are_rejected():
    assert CHECK.machine_reference_reasons("p = '" + "//" + "nas/share/data'")
    assert CHECK.machine_reference_reasons("p = 'smb:" + "//" + "nas/share'")
    # a protocol-relative public URL is stripped before the scan
    assert CHECK.machine_reference_reasons("see https:" + "//" + "github.com/org/repo") == []


def test_placeholder_and_math_cannot_hide_machine_paths():
    assert CHECK.machine_reference_reasons('p = "<' + "D" + ':\\\\data\\\\x>"')
    assert CHECK.machine_reference_reasons('p = "$' + "/" + 'home/user$"')
    assert CHECK.machine_reference_reasons("read from <repo>/data/file") == []
    assert CHECK.machine_reference_reasons(r"$\mathrm{ddof}=1$ and $p{0.5\columnwidth}$") == []
