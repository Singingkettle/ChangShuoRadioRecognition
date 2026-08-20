#!/usr/bin/env python
"""Validate one paper's contribution against docs/adding_a_new_paper.md.

Usage:  python tools/misc/check_paper.py <name>          # e.g. detection_is_easy
        python tools/misc/check_paper.py --list          # list paper dirs

Stdlib only (no mmengine needed except the optional `import csrr` purity probe,
which runs in a subprocess with mmcv blocked). HARD checks fail the run
(exit 1); WARN checks are reported but do not fail. Legacy papers are
grandfathered — running this on them documents debt, it does not gate them.
"""
import argparse
import ast
import py_compile
import re
import subprocess
import sys
from pathlib import Path

HARD, WARN = "HARD", "WARN"
_R = {"ok": "\033[32m", "no": "\033[31m", "wa": "\033[33m", "z": "\033[0m"}


def repo_root():
    p = Path(__file__).resolve()
    for up in [p, *p.parents]:
        if (up / "tools" / "train.py").exists() and (up / "csrr").is_dir():
            return up
    sys.exit("cannot find repo root (need tools/train.py + csrr/ above this script)")


class Report:
    def __init__(self):
        self.rows = []

    def add(self, name, level, ok, detail=""):
        self.rows.append((name, level, ok, detail))

    def failed(self):
        return any((not ok) and level == HARD for _, level, ok, _ in self.rows)

    def render(self):
        for name, level, ok, detail in self.rows:
            mark = f"{_R['ok']}PASS{_R['z']}" if ok else (
                f"{_R['no']}FAIL{_R['z']}" if level == HARD else f"{_R['wa']}WARN{_R['z']}")
            print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
        print()
        print("RESULT:", f"{_R['no']}FAILED{_R['z']}" if self.failed()
              else f"{_R['ok']}OK{_R['z']} (warnings do not gate)")


def read(p):
    return p.read_text(encoding="utf-8", errors="replace")


def base_targets(cfg_text):
    """Extract every string literal assigned to _base_ (scalar or list)."""
    try:
        tree = ast.parse(cfg_text)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_base_" for t in node.targets):
            v = node.value
            elts = v.elts if isinstance(v, (ast.List, ast.Tuple)) else [v]
            for e in elts:
                if isinstance(e, ast.Constant) and isinstance(e.value, str):
                    out.append(e.value)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    root = repo_root()
    cfgdir = root / "configs"

    if args.list or not args.name:
        papers = sorted(d.name for d in cfgdir.iterdir()
                        if d.is_dir() and d.name != "_base_")
        print("paper dirs:", " ".join(papers))
        return 0

    name = args.name
    d = cfgdir / name
    rp = Report()
    if not d.is_dir():
        print(f"no such paper dir: configs/{name}")
        return 2

    root_cfgs = sorted(f for f in d.glob("*.py") if f.name != "__init__.py")
    all_py = sorted(d.rglob("*.py"))
    mds = sorted(f for f in d.rglob("*.md") if not f.name.endswith("_zh-CN.md"))

    # 1. README pair
    has_readme = (d / "README.md").exists() and (d / "README_zh-CN.md").exists()
    rp.add("README.md + README_zh-CN.md exist", HARD, has_readme,
           "" if has_readme else "missing one of the pair")

    # 2. every .md has a bilingual pair + header switch links
    md_bad = []
    for md in mds:
        zh = md.with_name(md.stem + "_zh-CN.md")
        if not zh.exists():
            md_bad.append(f"{md.name}: no _zh-CN pair")
            continue
        if f"]({md.stem}_zh-CN.md)" not in read(md):
            md_bad.append(f"{md.name}: no link to Chinese page")
        if f"]({md.name})" not in read(zh):
            md_bad.append(f"{zh.name}: no link to English page")
    rp.add("all .md bilingual + header switch links", HARD, not md_bad,
           "; ".join(md_bad[:3]))

    # 3. # Paper: header on root configs (WARN per file; HARD: at least one)
    missing_hdr = [c.name for c in root_cfgs
                   if "# Paper:" not in "".join(read(c).splitlines(keepends=True)[:10])]
    with_hdr = len(root_cfgs) - len(missing_hdr)
    rp.add("at least one root config has '# Paper:'", HARD, with_hdr > 0 or not root_cfgs)
    if missing_hdr:
        rp.add("every root config has '# Paper:' (within head-10)", WARN, False,
               f"{len(missing_hdr)}/{len(root_cfgs)} without: {', '.join(missing_hdr[:4])}")

    # 4. filename hygiene (WARN). The doc's <method>_<modality>-<dataset>.py is a
    # soft target (modality-dependent; the mmdet detection cells legitimately
    # deviate), so machine-enforce only hygiene: lowercase, [a-z0-9_-], no spaces.
    pat = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.py$")
    bad_names = [c.name for c in root_cfgs if not pat.match(c.name)]
    if bad_names:
        rp.add("config filename hygiene (lowercase, no spaces)", WARN, False,
               f"{len(bad_names)} off: {', '.join(bad_names[:4])}")

    # 5. _base_ legal + resolves (dangling = HARD). A target is legal if it stays
    # inside this paper's own configs/<name>/ (intra-paper inheritance, incl. a
    # subdir's ../ back to the paper root — the jdm template does this) or inside
    # the shared configs/_base_/, or is an external mmdet:: reference. It may not
    # escape into another paper's dir or outside configs/.
    base_root, paper_root = str((cfgdir / "_base_").resolve()), str(d.resolve())
    dangling, illegal = [], []
    for c in all_py:
        for t in base_targets(read(c)):
            if t.startswith("mmdet::"):
                continue  # external package (mmdet exception) — reported by check 9
            tgt = str((c.parent / t).resolve())
            if not (tgt.startswith(paper_root) or tgt.startswith(base_root)):
                illegal.append(f"{c.name}→{t}")
                continue
            if not (c.parent / t).resolve().exists():
                dangling.append(f"{c.name}→{t}")
    rp.add("_base_ stays in configs/<name>/ or configs/_base_/ (or mmdet::)",
           HARD, not illegal, "; ".join(illegal[:3]))
    rp.add("_base_ targets resolve on disk", HARD, not dangling,
           "; ".join(dangling[:3]))

    # 6. no machine paths / hard parents[N]
    machine = []
    for c in all_py:
        txt = read(c)
        if "/home/" in txt:
            machine.append(f"{c.name}: /home/ path")
        if re.search(r"parents\[\d+\]", txt):
            machine.append(f"{c.name}: hard parents[N]")
    rp.add("no /home/ paths, no hard parents[N]", HARD, not machine,
           "; ".join(machine[:3]))

    # 7. py_compile every .py
    compile_bad = []
    for c in all_py:
        try:
            py_compile.compile(str(c), doraise=True)
        except py_compile.PyCompileError as e:
            compile_bad.append(f"{c.name}: {str(e).splitlines()[-1][:50]}")
    rp.add("py_compile all configs/<name>/*.py", HARD, not compile_bad,
           "; ".join(compile_bad[:3]))

    # 8. root README table rows: mirrored line, both columns → configs/<name>
    def find_row(fp):
        for i, ln in enumerate(read(fp).splitlines(), 1):
            if f"(configs/{name})" in ln:
                return i, ln
        return None, None
    en_i, en_ln = find_row(root / "README.md")
    zh_i, zh_ln = find_row(root / "README_zh-CN.md")
    row_ok = en_i is not None and en_i == zh_i
    both_cols = en_ln is not None and en_ln.count(f"(configs/{name})") >= 2
    rp.add("root READMEs: mirrored Supported-Methods row", HARD, row_ok,
           "" if row_ok else f"EN line {en_i} vs ZH line {zh_i}")
    rp.add("README row: both columns link configs/<name>", HARD, bool(both_cols),
           "" if both_cols else "a column links elsewhere (e.g. docs/)")

    # 9. core purity: runtime.txt MM-free + import csrr with mmcv blocked
    rt = read(root / "requirements" / "runtime.txt") if (
        root / "requirements" / "runtime.txt").exists() else ""
    mm_in_core = re.findall(r"(?im)^\s*(mmcv|mmdet|mmpretrain|mmsegmentation)\b", rt)
    rp.add("requirements/runtime.txt is MM-family-free", HARD, not mm_in_core,
           "found: " + ", ".join(sorted(set(mm_in_core))) if mm_in_core else "")
    probe = ("import sys,builtins\n_r=builtins.__import__\n"
             "def b(n,*a,**k):\n"
             " if n=='mmcv' or n.startswith('mmcv.'): raise ModuleNotFoundError('mmcv blocked')\n"
             " return _r(n,*a,**k)\n"
             "builtins.__import__=b\nimport csrr\nprint('ok')\n")
    try:
        r = subprocess.run([sys.executable, "-c", probe], cwd=str(root),
                           capture_output=True, text=True, timeout=120)
        csrr_ok = r.returncode == 0 and "ok" in r.stdout
        why = "" if csrr_ok else (r.stderr.strip().splitlines() or [""])[-1][:70]
    except Exception as e:  # noqa: BLE001
        csrr_ok, why = False, str(e)[:70]
    rp.add("import csrr works with mmcv blocked", HARD, csrr_ok, why)

    # 10. requirements/<name>.txt exact pins
    reqf = root / "requirements" / f"{name}.txt"
    if reqf.exists():
        loose = [ln.strip() for ln in read(reqf).splitlines()
                 if ln.strip() and not ln.strip().startswith("#")
                 and re.search(r"[A-Za-z0-9_.\-]", ln)
                 and "==" not in ln and not ln.strip().startswith("git+")
                 and not ln.strip().startswith("-")]
        rp.add(f"requirements/{name}.txt pins exact ==", HARD, not loose,
               "loose: " + ", ".join(loose[:3]) if loose else "")

    print(f"== check_paper: configs/{name} ({len(root_cfgs)} root configs, "
          f"{len(all_py)} .py, {len(mds)} .md) ==")
    rp.render()
    return 1 if rp.failed() else 0


if __name__ == "__main__":
    raise SystemExit(main())
