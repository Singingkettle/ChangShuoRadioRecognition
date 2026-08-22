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
AUTHOR_NAME, AUTHOR_EMAIL = "ChangShuo", "changshuo@bupt.edu.cn"
SCANNED_SUFFIXES = {
    ".py", ".pyi", ".sh", ".bash", ".ps1", ".bat", ".cmd", ".yaml", ".yml",
    ".json", ".toml", ".ini", ".cfg", ".conf", ".properties", ".md", ".rst",
    ".txt", ".csv", ".ipynb", ".html", ".htm", ".env", ".log",
}
# Suffix-less files that routinely carry machine-specific settings.
SCANNED_BASENAMES = {
    "Dockerfile", "Makefile", ".env", ".gitignore", ".gitattributes", ".dockerignore",
}
ALLOWED_REQUIREMENT_OPTIONS = (
    "--extra-index-url", "--index-url", "-i", "--find-links", "-f", "--no-binary",
    "--only-binary", "--prefer-binary", "--pre", "--trusted-host", "--use-feature",
    "--no-index",
)
REQUIREMENT_INCLUDE = re.compile(r"^(?:-r|--requirement|-c|--constraint)(?:\s+|=)(\S+)$")
# Exact means exact: no wildcard, no arbitrary equality, no compound specifier, no
# environment marker, no hash option on the same line.
EXACT_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)(?:\[[^]]+\])?==([^=\s,;*]+)$")
GIT_SHA_PIN = re.compile(
    r"^(?:([A-Za-z0-9][A-Za-z0-9_.-]*)\s*@\s*)?git\+\S+@[0-9a-fA-F]{40}(?:#\S*)?$")
# A placeholder or math span may not hide a machine path (see _strip_non_paths).
_HIDDEN_PATH = re.compile(
    r"(?<![A-Za-z0-9_])/(?:home|data|mnt|scratch|workspace|root|Users|tmp|opt|var|srv)(?:[/\\])"
    r"|(?<![A-Za-z0-9])[A-Za-z]:[/\\]|\\\\[A-Za-z0-9_.-]+[/\\]"
    r"|(?<![:A-Za-z0-9_./\\])//[A-Za-z0-9_.-]+/",
    re.IGNORECASE)
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
    """Drop public URLs, angle-bracket placeholders and $math$ before scanning.

    A placeholder or math span is removed only when its own content is not a
    machine path, so a drive letter, UNC root or machine-local POSIX root written
    inside the brackets or dollar signs is still scanned, while a placeholder-rooted
    relative path and ordinary LaTeX math stay allowed.
    """
    line = re.sub(r"https?://[^\s)>\]]+", "", line)

    def keep_if_path(match):
        return match.group(0) if _HIDDEN_PATH.search(match.group(1)) else ""

    line = re.sub(r"\S*<([^>\r\n]+)>\S*", keep_if_path, line)
    return re.sub(r"[$]([^$\r\n]*)[$]", keep_if_path, line)


def machine_reference_reasons(text):
    reasons = []
    for line_no, original in enumerate(text.splitlines(), 1):
        line = _strip_non_paths(original)
        # /usr is deliberately excluded: it collides with the #!/usr/bin/env
        # shebang. The rest are machine-local roots that must never be hard-coded.
        if re.search(r"(?<![A-Za-z0-9_])/(?:home|data|mnt|scratch|workspace|root|Users|tmp|opt|var|srv)(?:[/\\])",
                     line, re.IGNORECASE):
            reasons.append(f"line {line_no}: POSIX machine path")
        # Recognize both literal UNC text and its common source-escaped form.
        # Matching separator multiplicity prevents generated LaTeX command
        # pairs from being mistaken for network paths.
        drive_path = re.search(r"(?<![A-Za-z0-9])[A-Za-z]:[/\\]", line)
        literal_unc = re.search(
            r"(?<!\\)\\\\[A-Za-z0-9_.-]+\\(?!\\)[A-Za-z0-9_.-]+", line)
        escaped_unc = re.search(
            r"(?<!\\)\\\\\\\\[A-Za-z0-9_.-]+\\\\(?!\\)[A-Za-z0-9_.-]+", line)
        # POSIX-style network root (two leading slashes, no URL scheme in front).
        posix_unc = re.search(r"(?<![:A-Za-z0-9_./\\])//[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", line)
        if drive_path or literal_unc or escaped_unc or posix_unc:
            reasons.append(f"line {line_no}: Windows/UNC absolute path")
        if re.search(r"\b(?:smb|cifs|nfs|afp|sftp|ssh|file)://[A-Za-z0-9]", line):
            reasons.append(f"line {line_no}: network file URL")
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
        scanned = path.suffix.lower() in SCANNED_SUFFIXES or path.name in SCANNED_BASENAMES
        if not scanned or not path.is_file():
            continue
        findings.extend(f"{path}: {reason}" for reason in machine_reference_reasons(read(path)))
    return findings


def canonical_package_name(value):
    return re.sub(r"[-_.]+", "-", str(value).strip().lower())


def _requirement_line(raw):
    """Strip a pip comment: ``#`` starts one at line start or after whitespace."""
    return re.sub(r"(?:^|\s)#.*$", "", raw).strip()


def exact_pin_name(spec):
    """Canonical package name when ``spec`` is an exact pin, else ``None``.

    Exact means ``name==x.y.z`` (no ``*`` wildcard, no ``===``, no compound
    specifier, no environment marker) or ``name @ git+<url>@<40-hex-sha>``.
    """
    match = EXACT_PIN.match(spec)
    if match:
        return canonical_package_name(match.group(1))
    match = GIT_SHA_PIN.match(spec)
    if match and match.group(1):
        return canonical_package_name(match.group(1))
    return None


def iter_requirement_specs(root, path, _seen=None):
    """Yield ``(file, kind, text)`` over a requirements file and its in-repo includes.

    ``kind`` is ``spec``, ``option``, ``editable`` or ``bad-include``. ``-r``/``-c``
    includes are followed only while they stay inside the repository, so a loose
    pin cannot hide in an included file.
    """
    root, path = Path(root).resolve(), Path(path).resolve()
    seen = set() if _seen is None else _seen
    if path in seen or not path.is_file():
        return
    seen.add(path)
    for raw in read(path).splitlines():
        line = _requirement_line(raw)
        if not line:
            continue
        include = REQUIREMENT_INCLUDE.match(line)
        if include:
            target = (path.parent / include.group(1)).resolve()
            if not is_within(target, root) or not target.is_file():
                yield path, "bad-include", line
            else:
                yield from iter_requirement_specs(root, target, seen)
        elif line.startswith(("-e", "--editable")):
            yield path, "editable", line
        elif line.startswith("-"):
            yield path, "option", line
        else:
            yield path, "spec", line


def pinned_requirement_names(text):
    """Normalized names of exact pins in one requirements text (no include walking)."""
    names = set()
    for raw in text.splitlines():
        line = _requirement_line(raw)
        if line and not line.startswith("-"):
            name = exact_pin_name(line)
            if name:
                names.add(name)
    return names


def pinned_requirement_names_from_file(root, path):
    """Normalized names of exact pins, following in-repository ``-r``/``-c`` includes."""
    names = set()
    for _, kind, text in iter_requirement_specs(root, path):
        name = exact_pin_name(text) if kind == "spec" else None
        if name:
            names.add(name)
    return names


def runtime_check_problems(root, argv):
    """Policy for ``runtime_check``: ``{python}`` followed by an existing in-repo
    ``.py`` script. Interpreter flags such as ``-c``/``-m`` and other executables
    are rejected so the shell-free gate cannot be turned into a shell."""
    if not (isinstance(argv, list) and argv
            and all(isinstance(item, str) and item for item in argv)):
        return ["runtime_check must be a non-empty argv array of strings"]
    problems = []
    if argv[0] != "{python}":
        problems.append("argv[0] must be the {python} placeholder")
    if len(argv) < 2 or not argv[1].endswith(".py"):
        problems.append("argv[1] must be a repository .py script (no -c/-m)")
    for token in argv[1:]:
        if token.endswith(".py"):
            try:
                target = repo_relative_path(root, token)
            except (TypeError, ValueError) as exc:
                problems.append(f"{token}: {exc}")
                continue
            if not target.is_file():
                problems.append(f"{token}: missing")
    return problems


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
    if re.search(r"(?i)co-authored-by\s*:", message):
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

    runtime_bad = runtime_check_problems(root, manifest.get("runtime_check"))
    report.add("runtime_check is {python} plus an existing in-repo script (no -c/-m/shell)",
               HARD, not runtime_bad, "; ".join(runtime_bad[:3]))
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
    exception_packages = set()
    if exception_ok:
        for item in exceptions:
            if not (isinstance(item, dict) and all(
                    isinstance(item.get(key), str) and item.get(key).strip()
                    for key in ("package", "scope", "reason"))):
                exception_ok = False
                break
            exception_packages.add(canonical_package_name(item["package"]))
            try:
                scope = repo_relative_path(root, item["scope"])
            except (TypeError, ValueError):
                exception_ok = False
                break
            if not scope.exists():
                exception_ok = False
                break
    report.add("external framework exceptions are structured", HARD, exception_ok)
    req_names = set()
    try:
        requirement_path = repo_relative_path(root, manifest.get("requirements"))
        if requirement_path.is_file():
            req_names = pinned_requirement_names_from_file(root, requirement_path)
    except (TypeError, ValueError):
        pass
    missing_external_pins = sorted(exception_packages - req_names)
    report.add("declared external frameworks are exactly pinned in requirements", HARD,
               exception_ok and not missing_external_pins,
               ", ".join(missing_external_pins))

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


def _check_base_targets(root, name, manifest, report):
    cfg_root, paper_root = Path(root) / "configs", Path(root) / "configs" / name
    shared_root, illegal, dangling = cfg_root / "_base_", [], []
    declared_external = {
        canonical_package_name(item.get("package"))
        for item in (manifest or {}).get("external_framework_exceptions", [])
        if isinstance(item, dict) and item.get("package")
    }
    for config in paper_root.rglob("*.py"):
        for target in base_targets(read(config)):
            external = re.match(r"^([A-Za-z0-9_.-]+)::", target)
            if external:
                namespace = canonical_package_name(external.group(1))
                if namespace in declared_external:
                    continue
                illegal.append(f"{config.name}->{target} (undeclared external namespace)")
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
    for source, kind, text in iter_requirement_specs(root, req):
        if kind == "option":
            if not text.startswith(ALLOWED_REQUIREMENT_OPTIONS):
                loose.append(f"{source.name}: unsupported option {text}")
        elif kind == "bad-include":
            loose.append(f"{source.name}: include missing or outside repository: {text}")
        elif kind == "editable":
            loose.append(f"{source.name}: editable install {text}")
        elif exact_pin_name(text) is None and not GIT_SHA_PIN.match(text):
            loose.append(f"{source.name}: {text}")
    report.add("paper requirements use exact versions or immutable git SHAs (includes followed)",
               HARD, not loose, "; ".join(loose[:3]))


def validate_static(root, name, run_import_probe=True):
    root, report = Path(root).resolve(), Report("STATIC")
    paper_dir = root / "configs" / name
    report.add(f"configs/{name} exists", HARD, paper_dir.is_dir())
    if not paper_dir.is_dir():
        return report, None
    manifest = load_manifest(root, name, report)
    _check_manifest(root, name, manifest, report)
    _check_markdown_pairs(root, name, report)
    _check_base_targets(root, name, manifest, report)
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
    return report, manifest


def _git(root, *args):
    return subprocess.run(
        ["git", *args], cwd=str(root), capture_output=True, text=True, shell=False)


def _commit_records(root, revision_range):
    """One ``git show`` per commit with NUL-separated fields and the message last,
    so control characters inside a message can neither split a record nor hide a
    trailer from the policy check."""
    listing = _git(root, "rev-list", revision_range)
    if listing.returncode:
        return [], listing.stderr.strip()
    records = []
    for sha in listing.stdout.split():
        shown = _git(root, "show", "-s", "--format=%an%x00%ae%x00%cn%x00%ce%x00%B", sha)
        fields = shown.stdout.split("\x00", 4)
        if shown.returncode or len(fields) != 5:
            return [], f"unparseable commit record {sha[:8]}"
        records.append({
            "sha": sha, "author_name": fields[0], "author_email": fields[1],
            "committer_name": fields[2], "committer_email": fields[3],
            "message": fields[4],
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
    report.add("commit records are parseable", HARD, not record_error, record_error)
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
    problems = runtime_check_problems(root, argv)
    if problems:
        raise ValueError("; ".join(problems))
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
