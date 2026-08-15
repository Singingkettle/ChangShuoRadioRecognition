# Adding a New Paper's Code

English | [简体中文](adding_a_new_paper_zh-CN.md)

This page is how a new paper's code is added to ChangShuoRadioRecognition (CSRR).

**The algorithm short name is the directory name under `configs/`.** That is the
only name for the method in this repo. The branch name, both README columns, and
any optional `scripts/` follow it. Do not invent a second name, and do not
rename a directory that is already in the tree.

Rule: **one paper = `configs/<name>/` (configs + English/Chinese README pair +
optional `scripts/`) + native `csrr/` modules when needed + one row in each
root README.** Do not invent a new top-level directory (`projects/` and the
like). Do not add `tools/<name>/` or `docs/<name>/` for a new paper.

Train and test always enter through the shared scripts:

```bash
python tools/train.py configs/<name>/<config>.py
python tools/test.py configs/<name>/<config>.py <checkpoint.pth>
```

Examples:

- **JDM**: `configs/jdm/README.md` + root configs + `configs/jdm/scripts/`
  (merge checkpoints, rasterize figures, precompute proposals, and other
  paper-specific steps).
- **A typical AMC method** (CNN2): only `configs/cnn2/*.py` + the README pair.
  No `scripts/`.
- **DetectionIsEasy**: `tools/detection_is_easy/` + `docs/detection_is_easy/`
  is a **historical exception. Do not copy it.** New papers do not use that
  triad.

## Documentation language

Every user-facing document is a pair. **English is the default file.**

- English: `foo.md`
- Chinese: `foo_zh-CN.md` in the same directory
- Header links: `English | [简体中文](foo_zh-CN.md)` on the English page, and
  `[English](foo.md) | 简体中文` on the Chinese page

A new paper must ship `configs/<name>/README.md` and
`configs/<name>/README_zh-CN.md`. Do not put the method notes under
`docs/<name>/`.

## 1. configs/<name>/ — configs, notes, optional scripts

- One folder per paper. **Folder name = short name** (lowercase, underscores).
  Do not change it later. **Each config file is one experiment.**
- Naming: `<method>_<modality>-<dataset>.py` (for example
  `cnn2_iq-deepsig-201610A.py`, `jdm-det_fft-csrd.py`). Variants that actually
  reproduce the paper numbers may live under `experiments/` as
  `<main-config>_<suffix>.py`.
- **`experiments/` is reproduction mainline only** (paper-protocol evals and
  the operating point you actually report). Failed lr / epoch / seed / EMA /
  SWA variants, manifests, and goal files stay out of git.
- Every config (at least every root config) starts with:

  ```python
  # <one or two lines about the method>
  # Paper: "<title>", <venue> (<year or under review>).
  ```

- `_base_` may only point at `./xxx.py` in the same folder, shared
  `../_base_/...`, or an external package `mmdet::...`. **No**
  `../<other-paper>/`, absolute paths, or paths outside the repo.
- `data_root` / `work_dir` / `ann_file` are repo-relative (`data/...`,
  `work_dirs/...`). **No** `/home/<user>/...` machine paths.

### README.md and README_zh-CN.md

Notes live only in the README pair under `configs/<name>/`. Use this outline:

1. Title: `# <display name> — <paper title>`
2. Blockquote citation (authors, title, venue, year / under review, DOI / arXiv)
3. `## Method in one paragraph`
4. `## Paper section → code map`: `| paper | code |`
5. `## Data`: where the data comes from, on-disk layout, regenerate command if
   heavy assets are not in git; document any split that differs from the public
   protocol (DeepSig uses 50/10/40 here)
6. `## Train / evaluate`: one numbered bash block. Entry points must be
   `tools/train.py` / `tools/test.py`. Paper-specific steps use
   `python configs/<name>/scripts/...`
7. `## Results`: measured vs published numbers, with seed / error-bar policy
8. `## Documented deviations / notes`

Reproduction means: the official `configs/<name>/` runs should not sit far from
the published numbers. Do not invent an `amr_benchmark` (or similar) side
folder to stand for “a group of algorithms.”

Do not check in siege logs (`retune_campaign.md`, `retune_results.md`,
`goal_mode.md`). Put the main table and the deviations in the README.

### scripts/ (optional)

Create `configs/<name>/scripts/` only when this paper has unique steps: merging
two module checkpoints, drawing paper figures, precomputing proposals. Keep the
directory flat. Do not nest another `tools/`.

- Scripts use repo-relative paths. Find the repo root by walking up until
  `tools/train.py` and `csrr/` exist. Do not hard-code `parents[N]` or
  `/home/<user>/...`.
- If the paper needs a stack the core framework does not (for example mmdet),
  plugins may live here too:
  - configs use a bare module name in
    `custom_imports = dict(imports=['<module>'], allow_failed_imports=False)`
  - the caller inserts `scripts/` on `sys.path` before `Config.fromfile`
- Every `import` must resolve in-repo or in requirements.
- Extra deps go in `requirements/<name>.txt` (see
  `requirements/detection_is_easy.txt`), with a header comment on purpose and
  install prerequisites.

Ordinary classifiers share `tools/train.py` / `tools/test.py`. **Do not**
create an empty `scripts/` for them.

## 2. csrr/ — native framework modules

- New backbone → `csrr/models/backbones/<name>.py`,
  `@BACKBONES.register_module()`, subclass `BaseBackbone`, `forward` returns
  `(x,)`. New head → `csrr/models/heads/`, implement `forward` / `loss` /
  `predict` on `DataSample`. New dataset → `csrr/datasets/`,
  `@DATASETS.register_module()`, set `METAINFO = {'classes': (...)}`.
- Add the import and the class name to the matching `__init__.py`. Two traps:
  1. Those `__init__.py` files use **CRLF**. Edit at byte level; do not rewrite
     the whole file (that produces a fake full-file diff).
  2. If the last `__all__` entry has no trailing comma, appending a name
     silently concatenates strings (`'a' 'b'` → `'ab'`). Confirm the previous
     item has a comma.
- Keep only reusable model code here. Paper-specific glue goes in
  `configs/<name>/scripts/`, not `tools/<name>/`.

## 3. tools/ — shared entry points, not per-paper folders

- **Train:** `tools/train.py`
- **Test:** `tools/test.py` (classification collects `pred_score` into
  `paper.pkl`; detection / joint configs use mmengine `Runner.test()`)
- Shared extras may stay in `tools/analyze.py`, `tools/convert_datasets/`,
  `tools/misc/`
- **Do not** add `tools/<name>/` for a new paper
- `tools/detection_is_easy/` is a historical exception. Do not copy it.

## 4. Root README.md + README_zh-CN.md

- Add one alphabetical row under `## Supported Methods`:
  `| [<display>](configs/<name>) | [<paper title>](configs/<name>) |`
- The display name may be the paper's usual spelling (`JDM`), but **both
  columns link to `configs/<name>`**.
- The two READMEs are line-for-line mirrors: same insertion line, both files,
  same line numbers.
- Both files are CRLF. Preserve line endings.

## 5. Commit and PR

- Branch name `paper/<name>`, matching the `configs/` directory.
- **The only contributor is [Singingkettle](https://github.com/Singingkettle).**
  git author/committer is `ChangShuo <changshuo@bupt.edu.cn>` for that account.
  **No `Co-authored-by:` lines.** Do not put Cursor, an assistant, or another
  GitHub account in contributors, PR author, or commit metadata.
- One-line commit message, no conventional-commit prefix (follow `main`).
  Prefer **one clean commit** per paper (`--amend` + `--force-with-lease`
  during iteration; do not rewrite history after review).
- Keep the PR a **pure add**: do not touch unrelated existing code. The diff
  must not contain files that are not this paper.

## 6. Do not check in

- Manuscript files: `.tex` / `.pdf` / `.bib` / reviewer replies / figure-source
  PDFs (plot **scripts** are fine; finished figures are not)
- Datasets and heavy assets (memmap, npz caches, checkpoints) — ship a
  regenerate script, not the bytes
- Machine absolute paths, private server details, secrets
- One-off exploration (`build_*` / `aggregate_*` / `audit_*`, keepalive, sweep
  orchestrators) and failed siege configs — mainline only
- `amr_benchmark`, `tools/<name>/`, `docs/<name>/` side paths (DetectionIsEasy
  historical exception only)

## 7. Pre-merge checklist

```bash
# Syntax: every new .py compiles
python -m py_compile configs/<name>/*.py
# If the paper has scripts:
python -m py_compile configs/<name>/scripts/*.py

# Dead references (must be zero hits on new files)
grep -rE '<old-project-name>|/home/' configs/<name>

# Registry __init__ is valid (no implicit string concat; import present)
python -c "import ast; ast.parse(open('csrr/datasets/__init__.py').read())"

# Dual root READMEs share line numbers; both columns link to configs/<name>
grep -n '<display>' README.md README_zh-CN.md
```

- On the server: every config loads with `Config.fromfile`; native models build
  from the registry and run forward / backward; if you can train, run at least
  one epoch.
- After merge, grep `main` once more.
