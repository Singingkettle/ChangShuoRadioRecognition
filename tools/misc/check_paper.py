#!/usr/bin/env python
"""Validate a paper contribution without overstating what was checked.

Stages: STATIC (repository), PRE-MERGE (Git), RUNTIME (manifest command).
Experimental provenance and manuscript/PDF claims are external evidence and
are always reported as EVIDENCE: NOT RUN by this public-repository checker.
"""
from __future__ import annotations

import argparse
import ast
import json
import py_compile
import re
import subprocess
import sys
from pathlib import Path

HARD, WARN = "HARD", "WARN"
REPRODUCTION_LEVELS = {"exact", "statistical", "pipeline_only"}
MM_FAMILY = ("mmcv", "mmdet", "mmpretrain", "mmseg", "mmsegmentation")
AUTHOR_NAME, AUTHOR_EMAIL = "ChangShuo", "changshuo@bupt.edu.cn"
SCANNED_SUFFIXES = {
    ".py", ".sh", ".bash", ".ps1", ".bat", ".cmd", ".yaml", ".yml",
    ".json", ".toml", ".ini", ".cfg", ".md", ".txt", ".csv",
}
FORBIDDEN_ARTIFACT_SUFFIXES = {
    ".tex", ".pdf", ".bib", ".bbl", ".blg", ".aux", ".pth", ".pt", ".ckpt",
    ".onnx", ".npy", ".npz", ".memmap", ".jsonl",
}
REQUIRED_MANIFEST_FIELDS = {
    "schema_version", "name", "paper", "official_configs", "build_configs",
    "runtime_check", "requirements", "reproduction_level",
    "known_limitations", "external_framework_exceptions",
    "declared_core_changes",
}
_R = {"ok": "\033[32m", "no": "\033[31m", "wa": "\033[33m", "z": "\033[0m"}


class Report:
    def __init__(self, stage):
        self.stage, self.rows = stage, []

    def add(self, name, level, ok, detail=""):
        self.rows.append((name, level, bool(ok), str(detail)))

    def failed(self):
        return any(not ok and level == HARD for _, level, ok, _ in self.rows)

    def render(self):
        print(f"== {self.stage} gate ==")
        for name, level, ok, detail in self.rows:
            if ok:
                mark = f"{_R['ok']}PASS{_R['z']}"
            elif level == HARD:
                mark = f"{_R['no']}FAIL{_R['z']}"
            else:
                mark = f"{_R['wa']}WARN{_R['z']}"
            print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
        result = "FAILED" if self.failed() else "OK"
        color = _R["no"] if self.failed() else _R["ok"]
        print(f"{self.stage} RESULT: {color}{result}{_R['z']}\n")


def read(path):
    return Path(path).read_text(encoding="utf-8", errors="replace")


def repo_root():
    current = Path(__file__).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "tools" / "train.py").exists() and (candidate / "csrr").is_dir():
            return candidate
    raise SystemExit("cannot find repo root (need tools/train.py + csrr/)")


def is_within(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def repo_relative_path(root, value):
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError(f"absolute path: {value}")
    resolved = (Path(root) / candidate).resolve()
    if not is_within(resolved, root):
        raise ValueError(f"path escapes repository: {value}")
    return resolved


def base_targets(config_text):
    try:
        tree = ast.parse(config_text)
    except SyntaxError:
        return []
    targets = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(item, ast.Name) and item.id == "_base_"
                   for item in node.targets):
            continue
        values = node.value.elts if isinstance(node.value, (ast.List, ast.Tuple)) else [node.value]
        for value in values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                targets.append(value.value)
    return targets


def _strip_non_paths(line):
    line = re.sub(r"https?://[^\s)>\]]+", "", line)
    line = re.sub(r"\S*<[^>\r\n]+>\S*", "", line)
    return re.sub(r"[$][^$\r\n]*[$]", "", line)


def machine_reference_reasons(text):
    reasons = []
    for line_no, original in enumerate(text.splitlines(), 1):
        line = _strip_non_paths(original)
        # /usr is deliberately excluded: it collides with the #!/usr/bin/env
        # shebang. The rest are machine-local roots that must never be hard-coded.
        if re.search(r"(?<![A-Za-z0-9_])/(?:home|data|mnt|scratch|workspace|root|Users|tmp|opt|var|srv)(?:[/\\])",
                     line, re.IGNORECASE):
            reasons.append(f"line {line_no}: POSIX machine path")
        if re.search(r"(?<![A-Za-z0-9])(?:[A-Za-z]:[/\\]|\\\\[A-Za-z0-9_.-]+[/\\])", line):
            reasons.append(f"line {line_no}: Windows/UNC absolute path")
        if re.search(r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|"
                     r"172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b", line):
            reasons.append(f"line {line_no}: private IPv4 address")
        if re.search(r"[A-Za-z0-9_.-]+@(?:10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)", line):
            reasons.append(f"line {line_no}: user@private-ip endpoint")
        if re.search(r"\bparents\[\d+\]", line):
            reasons.append(f"line {line_no}: hard-coded parents[N]")
    return reasons


def scan_machine_references(paths):
    findings = []
    for path in sorted(Path(item) for item in paths):
        if path.suffix.lower() not in SCANNED_SUFFIXES or not path.is_file():
            continue
        findings.extend(f"{path}: {reason}" for reason in machine_reference_reasons(read(path)))
    return findings


def branch_is_valid(branch, name):
    return bool(re.fullmatch(
        rf"paper/{re.escape(name)}(?:-[a-z0-9][a-z0-9_-]*)?", branch))


def validate_commit_record(record):
    problems = []
    expected = (AUTHOR_NAME, AUTHOR_EMAIL)
    if (record["author_name"], record["author_email"]) != expected:
        problems.append("wrong author")
    if (record["committer_name"], record["committer_email"]) != expected:
        problems.append("wrong committer")
    message = record["message"].strip()
    if len(message.splitlines()) != 1:
        problems.append("commit message/body is not one line")
    if re.search(r"(?im)^Co-authored-by:", message):
        problems.append("Co-authored-by present")
    return problems


def undeclared_core_changes(changed_files, manifest):
    declared = {
        item.get("path") for item in manifest.get("declared_core_changes", [])
        if isinstance(item, dict)
    }
    return sorted(path for path in changed_files
                  if path.startswith("csrr/") and path not in declared)


def load_manifest(root, name, report):
    path = Path(root) / "configs" / name / "paper_manifest.json"
    if not path.exists():
        report.add("paper_manifest.json exists", HARD, False, str(path))
        return None
    try:
        manifest = json.loads(read(path))
    except (OSError, ValueError) as exc:
        report.add("paper_manifest.json parses", HARD, False, exc)
        return None
    report.add("paper_manifest.json parses", HARD, True)
    missing = sorted(REQUIRED_MANIFEST_FIELDS - set(manifest))
    report.add("manifest has all required fields", HARD, not missing, ", ".join(missing))
    return manifest


def _check_manifest(root, name, manifest, report):
    if manifest is None:
        return
    report.add("manifest schema_version is 1", HARD, manifest.get("schema_version") == 1)
    report.add("manifest name matches directory", HARD, manifest.get("name") == name)
    paper = manifest.get("paper")
    paper_ok = isinstance(paper, dict) and all(
        isinstance(paper.get(key), str) and paper.get(key).strip()
        for key in ("title", "venue", "status"))
    report.add("paper title/venue/status are present", HARD, paper_ok)

    official, builds = manifest.get("official_configs"), manifest.get("build_configs")
    report.add("official_configs is a non-empty list", HARD,
               isinstance(official, list) and bool(official))
    report.add("build_configs is a non-empty list", HARD,
               isinstance(builds, list) and bool(builds))
    official = official if isinstance(official, list) else []
    builds = builds if isinstance(builds, list) else []
    official_bad, header_bad, filename_bad, resolved_official = [], [], [], set()
    paper_dir = Path(root) / "configs" / name
    for value in official:
        try:
            path = repo_relative_path(root, value)
        except (TypeError, ValueError) as exc:
            official_bad.append(f"{value}: {exc}")
            continue
        if not is_within(path, paper_dir) or path.suffix != ".py" or not path.exists():
            official_bad.append(str(value))
            continue
        resolved_official.add(str(Path(value).as_posix()))
        if "# Paper:" not in "\n".join(read(path).splitlines()[:20]):
            header_bad.append(str(value))
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*\.py", path.name):
            filename_bad.append(str(value))
    report.add("official config paths exist and stay in paper directory", HARD,
               not official_bad, "; ".join(official_bad[:3]))
    report.add("every official config has # Paper: in first 20 lines", HARD,
               not header_bad, "; ".join(header_bad[:3]))
    report.add("official config filenames are lowercase", HARD,
               not filename_bad, "; ".join(filename_bad[:3]))
    build_bad = [str(value) for value in builds
                 if str(Path(value).as_posix()) not in resolved_official]
    report.add("build_configs is a subset of official_configs", HARD,
               not build_bad, "; ".join(build_bad[:3]))

    runtime = manifest.get("runtime_check")
    runtime_ok = (isinstance(runtime, list) and bool(runtime)
                  and all(isinstance(item, str) and item for item in runtime))
    runtime_path_bad = []
    if runtime_ok:
        for token in runtime[1:]:
            if token.endswith(".py"):
                try:
                    target = repo_relative_path(root, token)
                except (TypeError, ValueError) as exc:
                    runtime_path_bad.append(f"{token}: {exc}")
                    continue
                if not target.exists():
                    runtime_path_bad.append(token)
    report.add("runtime_check is an argv array with existing script paths", HARD,
               runtime_ok and not runtime_path_bad, "; ".join(runtime_path_bad[:3]))
    req = manifest.get("requirements")
    try:
        req_path = repo_relative_path(root, req)
        req_ok = req_path.exists() and req_path.is_file()
    except (TypeError, ValueError):
        req_ok = False
    report.add("manifest requirements path exists", HARD, req_ok, str(req))

    level = manifest.get("reproduction_level")
    report.add("reproduction_level is exact/statistical/pipeline_only", HARD,
               level in REPRODUCTION_LEVELS, str(level))
    limitations = manifest.get("known_limitations")
    limitations_ok = isinstance(limitations, list)
    if level in {"statistical", "pipeline_only"}:
        limitations_ok = limitations_ok and bool(limitations) and all(
            isinstance(item, str) and item.strip() for item in limitations)
    report.add("reproduction limitations match declared level", HARD, limitations_ok)

    exceptions = manifest.get("external_framework_exceptions")
    exception_ok = isinstance(exceptions, list)
    if exception_ok:
        for item in exceptions:
            if not (isinstance(item, dict) and all(
                    isinstance(item.get(key), str) and item.get(key).strip()
                    for key in ("package", "scope", "reason"))):
                exception_ok = False
                break
            try:
                scope = repo_relative_path(root, item["scope"])
            except (TypeError, ValueError):
                exception_ok = False
                break
            if not scope.exists():
                exception_ok = False
                break
    report.add("external framework exceptions are structured", HARD, exception_ok)

    core = manifest.get("declared_core_changes")
    core_bad = []
    if not isinstance(core, list):
        core_bad.append("declared_core_changes is not a list")
        core = []
    for item in core:
        if not isinstance(item, dict):
            core_bad.append(str(item))
            continue
        path_value, tests = item.get("path"), item.get("tests")
        try:
            path = repo_relative_path(root, path_value)
        except (TypeError, ValueError) as exc:
            core_bad.append(f"{path_value}: {exc}")
            continue
        if not str(Path(path_value).as_posix()).startswith("csrr/") or not path.exists():
            core_bad.append(str(path_value))
        if not isinstance(item.get("reason"), str) or not item["reason"].strip():
            core_bad.append(f"{path_value}: missing reason")
        if not isinstance(tests, list) or not tests:
            core_bad.append(f"{path_value}: missing tests")
            continue
        for test in tests:
            try:
                test_path = repo_relative_path(root, test)
            except (TypeError, ValueError) as exc:
                core_bad.append(f"{test}: {exc}")
                continue
            if not str(Path(test).as_posix()).startswith("tests/") or not test_path.exists():
                core_bad.append(str(test))
    report.add("declared core changes have reasons and existing tests", HARD,
               not core_bad, "; ".join(core_bad[:3]))


def _check_markdown_pairs(root, name, report):
    paper_dir = Path(root) / "configs" / name
    docs = sorted(path for path in paper_dir.rglob("*.md")
                  if not path.name.endswith("_zh-CN.md"))
    bad = []
    for path in docs:
        chinese = path.with_name(path.stem + "_zh-CN.md")
        if not chinese.exists():
            bad.append(f"{path.name}: missing Chinese pair")
            continue
        if f"]({path.stem}_zh-CN.md)" not in read(path):
            bad.append(f"{path.name}: missing Chinese switch link")
        if f"]({path.name})" not in read(chinese):
            bad.append(f"{chinese.name}: missing English switch link")
    report.add("all user-facing Markdown has bilingual switch-linked pairs",
               HARD, not bad, "; ".join(bad[:3]))


def _check_base_targets(root, name, report):
    cfg_root, paper_root = Path(root) / "configs", Path(root) / "configs" / name
    shared_root, illegal, dangling = cfg_root / "_base_", [], []
    for config in paper_root.rglob("*.py"):
        for target in base_targets(read(config)):
            if target.startswith("mmdet::"):
                continue
            resolved = (config.parent / target).resolve()
            if not (is_within(resolved, paper_root) or is_within(resolved, shared_root)):
                illegal.append(f"{config.name}->{target}")
            elif not resolved.exists():
                dangling.append(f"{config.name}->{target}")
    report.add("_base_ stays inside paper or shared _base_", HARD,
               not illegal, "; ".join(illegal[:3]))
    report.add("_base_ targets resolve on disk", HARD,
               not dangling, "; ".join(dangling[:3]))


def _check_root_readmes(root, name, report):
    def row(path):
        if not Path(path).exists():
            return None, None
        for number, line in enumerate(read(path).splitlines(), 1):
            if f"(configs/{name})" in line:
                return number, line
        return None, None
    en_number, en_line = row(Path(root) / "README.md")
    zh_number, _ = row(Path(root) / "README_zh-CN.md")
    report.add("root README rows are mirrored at the same line", HARD,
               en_number is not None and en_number == zh_number,
               f"English={en_number}, Chinese={zh_number}")
    report.add("both root README columns link to paper directory", HARD,
               en_line is not None and en_line.count(f"(configs/{name})") >= 2)


def _check_reproduction_claim(root, name, manifest, report):
    if manifest is None:
        return
    level = manifest.get("reproduction_level")
    english = read(Path(root) / "configs" / name / "README.md")
    chinese = read(Path(root) / "configs" / name / "README_zh-CN.md")
    token, zh_token = f"Reproduction level: `{level}`", f"复现等级：`{level}`"
    report.add("README pair declares manifest reproduction level", HARD,
               token in english and zh_token in chinese,
               f"expected '{token}' and '{zh_token}'")


def _check_requirements(root, manifest, report):
    if manifest is None:
        return
    try:
        req = repo_relative_path(root, manifest.get("requirements"))
    except (TypeError, ValueError):
        return
    if not req.exists():
        return
    loose = []
    for line in read(req).splitlines():
        value = line.strip()
        if not value or value.startswith(("#", "-")):
            continue
        if value.startswith("git+") and re.search(r"@[0-9a-fA-F]{40}(?:#|$)", value):
            continue
        if "==" not in value:
            loose.append(value)
    report.add("paper requirements use exact versions or immutable git SHAs",
               HARD, not loose, "; ".join(loose[:3]))


def _blocked_import_probe(root):
    probe = (
        "import builtins\n"
        f"blocked={repr(MM_FAMILY)}\n"
        "original=builtins.__import__\n"
        "def guarded(name,*args,**kwargs):\n"
        " root=name.split('.')[0]\n"
        " if root in blocked: raise ModuleNotFoundError(name+' blocked')\n"
        " return original(name,*args,**kwargs)\n"
        "builtins.__import__=guarded\n"
        "import csrr\nprint('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=str(root), capture_output=True,
        text=True, timeout=120, shell=False)
    detail = ""
    if result.returncode or "ok" not in result.stdout:
        detail = (result.stderr.strip().splitlines() or [result.stdout.strip() or "failed"])[-1]
    return result.returncode == 0 and "ok" in result.stdout, detail[:160]


def validate_static(root, name, run_import_probe=True):
    root, report = Path(root).resolve(), Report("STATIC")
    paper_dir = root / "configs" / name
    report.add(f"configs/{name} exists", HARD, paper_dir.is_dir())
    if not paper_dir.is_dir():
        return report, None
    manifest = load_manifest(root, name, report)
    _check_manifest(root, name, manifest, report)
    _check_markdown_pairs(root, name, report)
    _check_base_targets(root, name, report)
    compile_bad = []
    for path in sorted(paper_dir.rglob("*.py")):
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            compile_bad.append(f"{path.name}: {str(exc).splitlines()[-1]}")
    report.add("all paper Python files compile", HARD,
               not compile_bad, "; ".join(compile_bad[:3]))
    scan_paths = list(paper_dir.rglob("*"))
    if manifest is not None:
        try:
            scan_paths.append(repo_relative_path(root, manifest.get("requirements")))
        except (TypeError, ValueError):
            pass
    machine = scan_machine_references(scan_paths)
    report.add("paper contribution has no machine paths/private endpoints",
               HARD, not machine, "; ".join(machine[:3]))
    _check_root_readmes(root, name, report)
    _check_reproduction_claim(root, name, manifest, report)
    _check_requirements(root, manifest, report)
    runtime_req, mm_core = root / "requirements" / "runtime.txt", []
    if runtime_req.exists():
        mm_core = re.findall(
            r"(?im)^\s*(mmcv|mmdet|mmpretrain|mmseg|mmsegmentation)\b",
            read(runtime_req))
    report.add("core runtime requirements are MM-family-free", HARD,
               not mm_core, ", ".join(sorted(set(mm_core))))
    if run_import_probe:
        try:
            ok, detail = _blocked_import_probe(root)
        except Exception as exc:
            ok, detail = False, str(exc)
        report.add("import csrr works with all non-mmengine MM packages blocked",
                   HARD, ok, detail)
    return report, manifest


def _git(root, *args):
    return subprocess.run(
        ["git", *args], cwd=str(root), capture_output=True, text=True, shell=False)


def _commit_records(root, revision_range):
    fmt = "%H%x1f%an%x1f%ae%x1f%cn%x1f%ce%x1f%B%x1e"
    result = _git(root, "log", f"--format={fmt}", revision_range)
    if result.returncode:
        return [], result.stderr.strip()
    records = []
    for raw in result.stdout.split("\x1e"):
        raw = raw.strip("\r\n ")
        if not raw:
            continue
        fields = raw.split("\x1f", 5)
        if len(fields) == 6:
            records.append({
                "sha": fields[0], "author_name": fields[1],
                "author_email": fields[2], "committer_name": fields[3],
                "committer_email": fields[4], "message": fields[5],
            })
    return records, ""


def validate_premerge(root, name, base_ref, manifest):
    root, report = Path(root).resolve(), Report("PRE-MERGE")
    inside = _git(root, "rev-parse", "--is-inside-work-tree")
    report.add("running inside a Git worktree", HARD,
               inside.returncode == 0 and inside.stdout.strip() == "true",
               inside.stderr.strip())
    if inside.returncode:
        return report
    dirty = _git(root, "status", "--porcelain")
    report.add("worktree and index are clean", HARD,
               dirty.returncode == 0 and not dirty.stdout.strip(),
               " | ".join(dirty.stdout.strip().splitlines()[:3]))
    branch = _git(root, "branch", "--show-current").stdout.strip()
    report.add("branch matches paper/<name>[-topic]", HARD,
               branch_is_valid(branch, name), branch)
    base = _git(root, "rev-parse", "--verify", base_ref)
    report.add("base ref resolves", HARD, base.returncode == 0,
               base.stderr.strip() or base_ref)
    if base.returncode:
        return report
    merge_base = _git(root, "merge-base", base_ref, "HEAD")
    if merge_base.returncode:
        report.add("merge base resolves", HARD, False, merge_base.stderr.strip())
        return report
    revision_range = f"{merge_base.stdout.strip()}..HEAD"
    records, record_error = _commit_records(root, revision_range)
    report.add("paper branch contains commits after base", HARD, bool(records), record_error)
    commit_bad = []
    for record in records:
        commit_bad.extend(f"{record['sha'][:8]}: {problem}"
                          for problem in validate_commit_record(record))
    report.add("commit author/committer/message metadata is compliant", HARD,
               not commit_bad, "; ".join(commit_bad[:4]))
    diff_check = _git(root, "diff", "--check", revision_range)
    report.add("git diff --check is clean", HARD, diff_check.returncode == 0,
               " | ".join(diff_check.stdout.strip().splitlines()[:3]))
    # --name-status so that deletions are known: a file removed from a legacy side directory
    # (docs/<name>/, tools/<name>/) is the convention being applied and must not be flagged
    # as "the diff contains a side directory". Renames contribute their new path only.
    status_result = _git(root, "diff", "--name-status", revision_range)
    changed = []
    for line in status_result.stdout.splitlines():
        fields = line.strip().split("\t")
        if len(fields) < 2 or fields[0].startswith("D"):
            continue
        changed.append(fields[-1].strip().replace("\\", "/"))
    changed_machine = scan_machine_references(
        root / path for path in changed if (root / path).exists())
    report.add("changed text files have no machine paths/private endpoints",
               HARD, not changed_machine, "; ".join(changed_machine[:3]))
    forbidden = [
        path for path in changed
        if Path(path).suffix.lower() in FORBIDDEN_ARTIFACT_SUFFIXES
        or path.startswith((f"tools/{name}/", f"docs/{name}/", "projects/"))
    ]
    report.add("diff contains no manuscripts/heavy assets/side directories",
               HARD, not forbidden, "; ".join(forbidden[:4]))
    undeclared = undeclared_core_changes(changed, manifest or {})
    report.add("every changed csrr file is declared in manifest", HARD,
               not undeclared, "; ".join(undeclared[:4]))
    declared_test_bad = []
    for item in (manifest or {}).get("declared_core_changes", []):
        for test in item.get("tests", []):
            normalized = str(Path(test).as_posix())
            if normalized not in changed:
                declared_test_bad.append(normalized)
    report.add("declared core regression tests are part of the diff", HARD,
               not declared_test_bad, "; ".join(declared_test_bad[:4]))
    return report


def runtime_argv(root, manifest):
    if manifest is None:
        raise ValueError("manifest unavailable")
    argv = manifest.get("runtime_check")
    if not isinstance(argv, list) or not argv or not all(
            isinstance(item, str) and item for item in argv):
        raise ValueError("runtime_check must be a non-empty argv array")
    return [sys.executable if item == "{python}" else item for item in argv]


def validate_runtime(root, manifest, runner=subprocess.run):
    report = Report("RUNTIME")
    try:
        argv = runtime_argv(root, manifest)
    except ValueError as exc:
        report.add("runtime argv is valid", HARD, False, exc)
        return report
    report.add("runtime argv is valid and shell-free", HARD, True, repr(argv))
    try:
        result = runner(argv, cwd=str(root), shell=False)
    except Exception as exc:
        report.add("manifest runtime check exits zero", HARD, False, exc)
        return report
    report.add("manifest runtime check exits zero", HARD,
               result.returncode == 0, f"exit={result.returncode}")
    return report


def print_stage_summary(static_report, premerge_report, runtime_report):
    stages = {"STATIC": static_report, "PRE-MERGE": premerge_report, "RUNTIME": runtime_report}
    print("== gate summary ==")
    failed, all_ran = False, True
    for name, report in stages.items():
        if report is None:
            print(f"{name}: NOT RUN")
            all_ran = False
        else:
            print(f"{name}: {'FAILED' if report.failed() else 'OK'}")
            failed = failed or report.failed()
    print("EVIDENCE: NOT RUN (external experiment ledger + manuscript/PDF audit required)")
    print("OVERALL RESULT:", "FAILED" if failed else (
        "INCOMPLETE" if not all_ran else "REPOSITORY GATES OK; EVIDENCE STILL NOT RUN"))
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("name", nargs="?")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--pre-merge", action="store_true")
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--base-ref", default="origin/main")
    parser.add_argument("--repo-root", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve() if args.repo_root else repo_root()
    if args.list or not args.name:
        cfg_root = root / "configs"
        names = sorted(path.name for path in cfg_root.iterdir()
                       if path.is_dir() and path.name != "_base_")
        print("paper dirs:", " ".join(names))
        return 0
    static_report, manifest = validate_static(root, args.name)
    static_report.render()
    premerge_report = runtime_report = None
    if args.pre_merge or args.all:
        premerge_report = validate_premerge(root, args.name, args.base_ref, manifest)
        premerge_report.render()
    if args.runtime or args.all:
        runtime_report = validate_runtime(root, manifest)
        runtime_report.render()
    return print_stage_summary(static_report, premerge_report, runtime_report)


if __name__ == "__main__":
    raise SystemExit(main())
