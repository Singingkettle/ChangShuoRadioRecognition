# Adding a New Paper's Code

English | [简体中文](adding_a_new_paper_zh-CN.md)

This page is how a new paper's code is added to ChangShuoRadioRecognition (CSRR).

**The algorithm short name is the directory name under `configs/`.** That is the
only name for the method in this repo. The branch name, both README columns, and
any optional `scripts/` follow it. Do not invent a second name, and do not
rename a directory that is already in the tree.

Rule: **one paper = `configs/<name>/` (configs + English/Chinese README pair +
the non-config run files this paper needs) + native `csrr/` modules when a model
is missing + one row in each root README.** Do not invent a new top-level
directory (`projects/` and the like). Do not add `tools/<name>/` or
`docs/<name>/` for a new paper.

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
- **DetectionIsEasy**: the one **documented dependency exception** — its
  detection stage uses `mmdet`. That dependency is isolated to
  `requirements/detection_is_easy.txt` and is **never a core dependency**. New
  papers do not copy it; they implement natively (see §0).

## 0. Dependencies and framework scope (read first)

CSRR is built on **`mmengine` only**. That is deliberate.

- **`csrr/` core must import and run with `mmengine` alone.** No `csrr/`
  module may hard-import another MM-family package (`import mmcv`,
  `import mmdet`, …). Anything image-utility-like that MM-family used to
  provide is done with `cv2` / `PIL` / `numpy` instead.

- **The core install depends on `mmengine` (and PyTorch). Do not add another
  MM-family library** — `mmdet`, `mmcv`, `mmpretrain`, `mmsegmentation`, … The
  MM-family is large, tightly coupled, and version-fragile; pulling it in bloats
  the environment and breaks reproducibility across machines.
- **Core dependency versions are pinned and stay fixed.** Do not bump them for a
  new paper. If a paper truly needs a newer core, raise it separately.
- **A model this repo does not have is implemented natively under `csrr/`**
  (§2), registered in the CSRR registry. **Do not** reach for an external
  MM-family package to supply a backbone / head / detector. Native model files
  live under `csrr/models/…` and are **kept separate from a paper's scripts**.
- **Everything a paper needs to run, except its configs and its native model
  files, lives under `configs/<name>/`.** Nothing for a paper goes under
  `tools/<name>/`, `projects/`, or `docs/<name>/`.
- Paper-only extra dependencies may be pinned in `requirements/<name>.txt`, but
  **adding an MM-family library there is strongly discouraged**; it is reserved
  for the DetectionIsEasy mmdet exception, which stays isolated and optional.
- **Pin exact versions, not ranges.** The core pin is `mmengine==<pinned>`;
  paper extras pin `pkg==x.y.z` too. Range pins (`>=a,<b`) drift across
  machines and break byte-level reproduction.
- Requirements map: `requirements/runtime.txt` = what `setup.py` installs
  (**must stay MM-free**); `requirements/<name>.txt` = one paper's isolated
  extras; `mminstall.txt` is legacy and must not grow.

Repo map — every top-level directory and its single role:

| directory | role |
|---|---|
| `csrr/` | the framework: models, datasets, transforms, metrics, engine (§2) |
| `csrr/performance/` | paper-figure / results-summary module (figures, metrics tables) |
| `csrr/apis/` | inference entry points |
| `configs/_base_/` | shared base configs (datasets, schedules, runtimes) reused across papers |
| `configs/<name>/` | one paper: configs + README pair + its run files (§1) |
| `tools/` | shared train/test entry points only (§3) |
| `docs/` | framework-level docs only (install, getting started) + `docs/dataset/` shared dataset notes. **No per-paper docs.** |
| `tests/` | unit tests; a new metric in `csrr/evaluation/` should come with a test in `tests/test_evaluation/` |
| `requirements/` | dependency pins, per the map above |

## Documentation language

Every user-facing document is a pair. **English is the default file.**

- English: `foo.md`
- Chinese: `foo_zh-CN.md` in the same directory
- Header links: `English | [简体中文](foo_zh-CN.md)` on the English page, and
  `[English](foo.md) | 简体中文` on the Chinese page

A new paper must ship `configs/<name>/README.md` and
`configs/<name>/README_zh-CN.md`. Do not put the method notes under
`docs/<name>/`.

## 1. configs/<name>/ — configs, notes, and this paper's run files

- One folder per paper. **Folder name = short name** (lowercase, underscores).
  Do not change it later. **Each config file is one experiment.**
- Naming: `<method>_<modality>-<dataset>.py` (for example
  `cnn2_iq-deepsig-201610A.py`, `jdm-det_fft-csrd.py`). Variants that actually
  reproduce the paper numbers may live under `experiments/` as
  `<main-config>_<suffix>.py`.
- **`experiments/` is reproduction mainline only** (paper-protocol evals and
  the operating point you actually report). Failed lr / epoch / seed / EMA /
  SWA variants, manifests, and goal files stay out of git.
- Every root config carries this near the top — an Apache license block may
  precede it; keep it within the first few comment lines:

  ```python
  # <one or two lines about the method>
  # Paper: "<title>", <venue> (<year or under review>).
  ```

- `_base_` may point at (a) `./xxx.py` in the same folder; (b) another config
  **inside this paper's own `configs/<name>/`** — for example a variant under
  `configs/<name>/experiments/` that inherits the paper's root config via
  `../<root-config>.py` (the JDM template does exactly this); (c) a shared
  `../_base_/...`; or (d) an external `mmdet::...` **(mmdet only for the
  DetectionIsEasy exception)**. **No** `../<other-paper>/`, absolute paths, or
  paths outside the repo. Every `_base_` target must resolve on disk.
- `data_root` / `work_dir` / `ann_file` are repo-relative (`data/...`,
  `work_dirs/...`). **No** `/home/<user>/...` machine paths.

### Run files under configs/<name>/

The non-config files a paper needs to run — figure scripts, data-prep scripts,
a mmdet plugin module, an evaluation helper — live under `configs/<name>/`. Keep
them flat; do not nest another `tools/`. Native model code does **not** go here;
it goes under `csrr/` (§2).

- Scripts use repo-relative paths. **Find the repo root by walking up until
  `tools/train.py` and `csrr/` both exist. Do not hard-code `parents[N]` or
  `/home/<user>/...`.**
- Every `import` must resolve in-repo or in requirements.
- For the DetectionIsEasy mmdet exception, the plugin module lives here too:
  configs use a bare module name in
  `custom_imports = dict(imports=['<module>'], allow_failed_imports=False)`, and
  the caller inserts this directory on `sys.path` before `Config.fromfile`.
- Extra deps go in `requirements/<name>.txt` (see
  `requirements/detection_is_easy.txt`), with a header comment on purpose and
  install prerequisites. MM-family extras are discouraged (§0).

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
   `python configs/<name>/...`
7. `## Results`: measured vs published numbers, with seed / error-bar policy
8. `## Documented deviations / notes`

Reproduction means: the official `configs/<name>/` runs should not sit far from
the published numbers. Do not invent an `amr_benchmark` (or similar) side
folder to stand for “a group of algorithms.”

Do not check in siege logs (`retune_campaign.md`, `retune_results.md`,
`goal_mode.md`). Put the main table and the deviations in the README.

## 2. csrr/ — native framework modules (where any missing component goes)

If the paper needs a **framework component** CSRR does not have — not only a
model, but also a dataset, a special data-loading / preprocessing transform, a
sampler, a filter, a metric, a loss — **implement it natively in the matching
`csrr/` subpackage and register it.** Do not pull in an MM-family package for it
(§0), and **do not dump it into `configs/<name>/` next to the configs.** CSRR has
a full registry (see `csrr/registry.py`: `DATASETS`, `TRANSFORMS`, `METRICS`,
`MODELS`, `DATA_SAMPLERS`, `DATA_FILTERS`, …), so every reusable piece has a home.

Where each component type goes (CSRR subpackage → registry):

| you are adding | put it under | register with |
|---|---|---|
| model backbone | `csrr/models/backbones/` | `@MODELS` — subclass `BaseBackbone`, `forward` returns `(x,)` |
| detector / classifier | `csrr/models/{detectors,classifiers}/` | `@MODELS` |
| head (`forward` / `loss` / `predict` on `DataSample`) | `csrr/models/heads/` | `@MODELS` |
| loss | `csrr/models/losses/` | `@MODELS` |
| dataset | `csrr/datasets/` | `@DATASETS`, set `METAINFO = {'classes': (...)}` |
| **data-loading / preprocessing transform** | **`csrr/datasets/transforms/`** | `@TRANSFORMS` |
| sampler / filter | `csrr/datasets/{samplers,filters}/` | `@DATA_SAMPLERS` / `@DATA_FILTERS` |
| metric / evaluator | `csrr/evaluation/metrics/` | `@METRICS` |
| hook / optimizer / scheduler | `csrr/engine/` | `@HOOKS` / `@OPTIMIZERS` / `@PARAM_SCHEDULERS` |
| data structure | `csrr/structures/` | — |
| visualizer | `csrr/visualization/` | `@VISUALIZERS` |

`MODELS` is the single home for every model part: `csrr/models/builder.py`
aliases `BACKBONES`, `NECKS`, `HEADS`, `LOSSES`, and `CLASSIFIERS` to it, so
legacy code's `@BACKBONES.register_module()` and a fresh `@MODELS.register_module()`
land in the same registry — register with `@MODELS`. Paper figures/tables that
you want reusable can register in `csrr/performance/` (`@FIGURES` / `@TABLES`,
driven by `tools/analyze.py`), the way JDM ships its plots; a plain plot script
under `configs/<name>/` is also fine for one paper.

Do not cargo-cult empty scaffolding. Some declared names have nothing behind
them — an `ANALYSIS` registry pointing at a `csrr.analysis` module that does not
exist, near-empty corners of `csrr/apis/`. Put your component in the subpackage
whose registry is actually wired (the table above) and confirm
`register_all_modules()` imports it.

- A **custom dataset-loading method is a framework component**: it belongs in
  `csrr/datasets/transforms/` (`@TRANSFORMS`), not lumped into
  `configs/<name>/`. Same for a custom metric, sampler, or loss — each goes in
  its own `csrr/` subpackage, not mixed with the config files.
- These files stay under `csrr/`, **separate from the paper's `configs/<name>/`
  run files.**
- Add the import and the class name to the matching `__init__.py`. Two traps:
  1. Those `__init__.py` files use **CRLF**. Edit at byte level; do not rewrite
     the whole file (that produces a fake full-file diff).
  2. If the last `__all__` entry has no trailing comma, appending a name
     silently concatenates strings (`'a' 'b'` → `'ab'`). Confirm the previous
     item has a comma.
- Keep only reusable framework code here. **Paper-specific glue** — figure
  scripts, data-generation orchestration, and (for the DetectionIsEasy
  exception) the mmdet plugin module — goes in `configs/<name>/`, not
  `tools/<name>/`. The mmdet-exception plugin is the one case that lives in
  `configs/<name>/` rather than `csrr/`, because it is mmdet-registered, not a
  native CSRR component.

## 3. tools/ — shared entry points, not per-paper folders

- **Train:** `tools/train.py`
- **Test:** `tools/test.py` (classification collects `pred_score` into
  `paper.pkl`; detection / joint configs use mmengine `Runner.test()`)
- Shared extras may stay in `tools/analyze.py`, `tools/convert_datasets/`,
  `tools/misc/`
- **Do not** add `tools/<name>/` for a new paper. A paper's run files go under
  `configs/<name>/` (§1); its models go under `csrr/` (§2).

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
- **Pull the latest `main` before you push**, and rebase onto it. Keep the PR a
  **pure add**: do not touch unrelated existing code. The diff must not contain
  files that are not this paper.

## 6. Do not check in

- Another MM-family library in the **core** requirements (`mmdet`, `mmcv`,
  `mmpretrain`, …). Core stays `mmengine`-only (§0); the DetectionIsEasy mmdet
  exception is isolated to `requirements/detection_is_easy.txt`.
- Manuscript files: `.tex` / `.pdf` / `.bib` / reviewer replies / figure-source
  PDFs (plot **scripts** are fine; finished figures are not)
- Datasets and heavy assets (memmap, npz caches, checkpoints, prediction dumps,
  `.jsonl` diagnostics) — ship a regenerate script, not the bytes
- Machine absolute paths, private server details, secrets
- One-off exploration (`build_*` / `aggregate_*` / `audit_*`, keepalive, sweep
  orchestrators) and failed siege configs — mainline only
- `amr_benchmark`, `tools/<name>/`, `projects/`, `docs/<name>/` side paths

## 7. Pre-merge checklist

Run the validator first. It mechanizes every check below and must print
`RESULT: OK` before you open the PR:

```bash
python tools/misc/check_paper.py <name>
```

It fails (`RESULT: FAILED`, non-zero exit) on the HARD rules — README pair with
Chinese mirror + header switch links, at least one `# Paper:` header, `_base_`
legality and on-disk resolution, no `/home/` or hard `parents[N]` paths,
`py_compile`, mirrored root-README rows whose **both** columns link
`configs/<name>`, `import csrr` with `mmcv` blocked, and exact `==` pins in
`requirements/<name>.txt` — and only WARNs on soft ones (filename hygiene,
per-config `# Paper:` coverage). It is stdlib-only and runs with `mmengine`
alone. The commands below are what it automates, for checking a single rule by
hand:

```bash
# Syntax: every new .py compiles
python -m py_compile configs/<name>/*.py
# If the paper ships run files under configs/<name>/:
python -m py_compile configs/<name>/**/*.py

# Dead references (must be zero hits on new files)
grep -rE '<old-project-name>|/home/|parents\[' configs/<name>

# Registry __init__ is valid (no implicit string concat; import present)
python -c "import ast; ast.parse(open('csrr/datasets/__init__.py').read())"

# Core import works with mmengine alone (mmcv/mmdet absent or blocked)
python -c "import csrr"   # in an env with mmengine only
# Core stays mmengine-only: no MM-family in the core requirements
grep -riE 'mmcv|mmdet|mmpretrain' requirements/*.txt   # only detection_is_easy.txt may match

# Every .md ships a Chinese pair with the header switch link
ls configs/<name>/*_zh-CN.md

# Dual root READMEs share line numbers; both columns link to configs/<name>
grep -n '<display>' README.md README_zh-CN.md
```

- On the server: every config loads with `Config.fromfile`; native models build
  from the registry and run forward / backward; if you can train, run at least
  one epoch.
- After merge, grep `main` once more.
