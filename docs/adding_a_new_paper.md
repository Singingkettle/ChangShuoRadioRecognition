# Adding a New Paper's Code

English | [简体中文](adding_a_new_paper_zh-CN.md)

This page is how a new paper's code is added to ChangShuoRadioRecognition (CSRR).

**The algorithm short name is the directory name under `configs/`.** That is the
only name for the method in this repo. The branch name, both README columns, and
any optional `scripts/` follow it. Do not invent a second name, and do not
rename a directory that is already in the tree.

Rule: **one paper = `configs/<name>/` (manifest + configs + English/Chinese
README pair + the non-config run files this paper needs) + native `csrr/`
modules when a model is missing + one row in each root README.** Do not invent a new top-level
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
- **DetectionIsEasy**: its detector experiments and public runtime both use
  `mmdet`, `mmcv-lite`, and `mmengine`, pinned in
  `requirements/detection_is_easy.txt`. This is an ordinary, declared
  paper-level integration, not a claim that the detectors were reimplemented
  natively in CSRR.

## 0. Dependencies and framework scope (read first)

The base CSRR installation is PyTorch + `mmengine`. Papers may install external
frameworks, including `mmdet` and `mmcv`, when those packages are part of the
actual experimental implementation. Releasing a different implementation merely
to reduce dependencies is prohibited unless equivalence is independently shown.

- **Distinguish base and paper environments.** `requirements/runtime.txt` is
  the lightweight base installation. `requirements/<name>.txt` is the complete,
  pinned environment extension needed by one paper and may include MM-family
  packages. Installing that file is part of reproducing the paper, not an
  unsupported workaround.
- **The released path must match the measured path.** If reported experiments
  used MMDetection, the official configs, registry, preprocessing, evaluator,
  post-processing, and runtime check must use the documented MMDetection stack.
  Do not label an adapter as a native CSRR implementation, and do not substitute
  an unvalidated rewrite for the code that generated the reported numbers.
- **External frameworks are declared, not hidden.** List each one in the
  manifest's `external_framework_exceptions` (legacy field name), including its
  package, repository scope, and why it is required. The README must state the
  exact versions, OS/CUDA constraints, compiled-versus-lite variant, and every
  operator fallback that can alter predictions.
- **Pin the full compatibility tuple.** Record at least Python, PyTorch,
  torchvision, MMEngine, MMDetection/MMCV where used, NumPy, CUDA wheel source,
  and task evaluators. Exact package versions are necessary but not sufficient:
  runtime provenance also records the resolved versions and fallback flags.
- **Core dependency changes are deliberate.** A package may be promoted from a
  paper extra to `requirements/runtime.txt` only when shared CSRR modules need it,
  all supported installation instructions and CI environments are updated, and
  a clean base install is tested. Otherwise keep it paper-scoped.
- **Native implementation remains an option, not a dependency rule.** A missing
  component may be implemented under `csrr/` (§2), or integrated from the same
  external framework used in the experiments. A rewrite requires frozen-input
  forward/post-processing comparisons and seed-level metric equivalence before
  it can replace the experimental implementation.
- **Everything a paper needs to run, except its configs and its native model
  files, lives under `configs/<name>/`.** Nothing for a paper goes under
  `tools/<name>/`, `projects/`, or `docs/<name>/`.
- **Pin exact versions, not ranges.** The core pin is `mmengine==<pinned>`;
  paper extras pin `pkg==x.y.z` too. Range pins (`>=a,<b`) drift across
  machines and break byte-level reproduction.
- **Exact means exact.** `pkg==x.*`, `pkg===x`, compound specifiers,
  environment-marker lines and editable installs are not accepted as pins;
  `name @ git+<url>@<40-hex-sha>` is. `-r`/`-c` includes are followed only
  while they stay inside the repository, so a loose pin cannot hide in an
  included file.
- Requirements map: `requirements/runtime.txt` = what `setup.py` installs by
  default; `requirements/<name>.txt` = one paper's declared environment
  extension; `mminstall.txt` is legacy and must not grow.

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
  `../_base_/...`; or (d) an external framework target such as `mmdet::...`
  when that package is pinned and declared in the manifest. **No**
  `../<other-paper>/`, absolute paths, or paths outside the repo. Every local
  `_base_` target must resolve on disk; every external target must load in the
  clean-clone runtime gate.
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
- An external-framework plugin module lives here too: configs use a bare module name in
  `custom_imports = dict(imports=['<module>'], allow_failed_imports=False)`, and
  the caller inserts this directory on `sys.path` before `Config.fromfile`.
- Extra deps go in `requirements/<name>.txt` (see
  `requirements/detection_is_easy.txt`), with a header comment on purpose and
  install prerequisites. Document the exact framework variant and operator
  fallbacks, not only the top-level package version (§0).

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

### paper_manifest.json

Every new paper has `configs/<name>/paper_manifest.json`; use
`docs/paper_manifest.example.json` as the schema example. It is the source of
truth for the paper identity, official configs, build-gate configs, runtime
check, requirements, reproduction level, external-framework exceptions, and
declared core changes. This explicit list prevents run scripts from being
misclassified as configs.

Despite its legacy name, `external_framework_exceptions` is the authoritative
list of paper-level framework dependencies. Each `package` must match an exact
`package==version` entry in the manifest's requirements file; `scope` identifies
the repository subtree that integrates it; `reason` states why it is part of the
measured implementation. An external config namespace such as `mmdet::` is legal
only when that package is declared and pinned. Do not list a package merely to
bypass the path check.

Every path in the manifest is repo-relative, exists, and stays inside the
repository. Every official config has `# Paper:` near its head. A changed
`csrr/` file is listed in `declared_core_changes` with a concrete reason and at
least one existing regression test. `runtime_check` is an argv array; use the
special first token `{python}` for the active `sys.executable`. It is never a
shell command string.

### Reproduction contract and evidence ledger

`reproduction_level` has exactly one of these values:

- `exact`: the same data realization, config and artifacts can be regenerated
  within a declared numerical tolerance;
- `statistical`: the protocol is reproducible but the realization differs; set
  the statistical acceptance rule before running;
- `pipeline_only`: only the workflow is reproducible; do not claim that the
  published numbers are reproduced.

“Not far from the published numbers” is not an acceptance criterion.
`statistical` and `pipeline_only` require non-empty `known_limitations`, and the
README pair must state `Reproduction level: <level>`.

Every reported table keeps a private, append-only evidence ledger outside the
public repository. Preserve every seed-level value, split, metric, aggregation
formula (including sample vs population standard deviation), rounding rule and
the script that generated the table. Each result row maps to the commit SHA,
config hash, dataset manifest/checksum, environment, exact argv, checkpoint,
summary path and real archive location. Missing evidence is `na` plus a reason;
never put annotations such as `(archived)` inside a path field. Private server
paths belong only in this internal ledger, never in public code or README files.

Keep every failed run, learning-rate rescue and hedge run in the internal
tuning ledger. If detector/model families use different learning rates, say so
in the comparison. Claims using “all”, “only”, “essential”, “law” or “SOTA”
need a claim audit: define the criterion, report sample size and relevant
statistics, and list counterexamples. A trend is not a law merely because its
Pearson correlation is negative.

The manuscript remains outside this GitHub repository, but its release audit
is still mandatory. Data Availability must match what is actually released; if
the shard driver or seeds are missing, do not claim “reproduce every number”.
Build the PDF from a clean auxiliary-file state and inspect rendered figure
labels, captions, body text, code paths, page count and the venue’s current
official limit.

Do not invent an `amr_benchmark` (or similar) side folder to stand for “a group
of algorithms.”

Do not check in siege logs (`retune_campaign.md`, `retune_results.md`,
`goal_mode.md`). Put the main table and the deviations in the README.

## 2. csrr/ — native framework modules (where any missing component goes)

If a paper needs a reusable **native CSRR component** — a model, dataset,
transform, sampler, filter, metric, or loss — implement it in the matching
`csrr/` subpackage and register it. Paper-specific adapters for a declared
external framework stay under `configs/<name>/`; they are not represented as
native CSRR modules. CSRR has a full registry (see `csrr/registry.py`:
`DATASETS`, `TRANSFORMS`, `METRICS`, `MODELS`, `DATA_SAMPLERS`,
`DATA_FILTERS`, …), so every genuinely reusable native piece has a home.

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
  scripts, data-generation orchestration, and external-framework plugin modules
  — goes in `configs/<name>/`, not `tools/<name>/`. A plugin registered into
  MMDetection or another external registry is not a native CSRR component.

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

- Branch name is exactly `paper/<name>` or `paper/<name>-<topic>`.
- **The only contributor is [Singingkettle](https://github.com/Singingkettle).**
  git author/committer is `ChangShuo <changshuo@bupt.edu.cn>` for that account.
  **No `Co-authored-by:` lines.** Do not put Cursor, an assistant, or another
  GitHub account in contributors, PR author, or commit metadata.
- One-line commit message, no conventional-commit prefix (follow `main`).
  Prefer **one clean commit** per paper (`--amend` + `--force-with-lease`
  during iteration; do not rewrite history after review).
- **Pull the latest `main` before you push**, and rebase onto it. Keep the PR
  **scope-pure**: changes are limited to this paper and the reusable core pieces
  it genuinely requires. A modified shared core file needs a manifest
  declaration, a reason, a regression test, and evidence that the branch adds
  no test failure. Unrelated cleanup is a separate change.

## 6. Do not check in

- An undeclared or loosely pinned external framework dependency. Paper-scoped
  `mmdet`/`mmcv` dependencies are allowed, but they must be in
  `requirements/<name>.txt`, the manifest, the README, and the clean-clone gate;
  shared-runtime promotion follows §0.
- Manuscript files: `.tex` / `.pdf` / `.bib` / reviewer replies / figure-source
  PDFs (plot **scripts** are fine; finished figures are not)
- Datasets and heavy assets (memmap, npz caches, checkpoints, prediction dumps,
  `.jsonl` diagnostics) — ship a regenerate script, not the bytes
- Machine absolute paths, private server details, secrets
- One-off exploration (`build_*` / `aggregate_*` / `audit_*`, keepalive, sweep
  orchestrators) and failed siege configs — mainline only
- `amr_benchmark`, `tools/<name>/`, `projects/`, `docs/<name>/` side paths

## 7. Three release gates

No single green line proves a paper is releasable. The validator reports each
repository gate separately and always reports the external evidence gate as
`NOT RUN`.

### Gate A — static repository and Git

```bash
python tools/misc/check_paper.py <name>
python tools/misc/check_paper.py <name> --pre-merge --base-ref origin/main
```

The static gate validates the manifest, bilingual docs, config headers and
paths, syntax, exact dependency pins, external-framework declarations, machine paths,
private endpoints and README consistency. The requirements scan follows
in-repository `-r`/`-c` includes and rejects wildcard, arbitrary-equality,
compound and editable specifiers. `runtime_check` must be `{python}` followed by
an existing in-repository `.py` script; interpreter flags (`-c`, `-m`) and other
executables are rejected so the shell-free gate cannot become a shell. The
pre-merge gate validates the branch, author/committer, one-line messages,
absence of `Co-authored-by` anywhere in the message (commit records are read one
commit at a time, so control characters in a message cannot hide a trailer),
`git diff --check`, forbidden artifacts, and declared core changes/tests.

The machine-path scan covers machine-local POSIX roots — `/home`, `/data`,
`/mnt`, `/scratch`, `/workspace`, `/root`, `/Users`, `/tmp`, `/opt`, `/var`,
`/srv` (`/usr` is excluded so a `#!/usr/bin/env` shebang is not a finding) —
Windows drive/UNC paths, POSIX-style `//<host>/<share>` roots, `smb://` and
`file://`-style network URLs, private IPv4 endpoints and hard `parents[N]`.
Suffix-less configuration files (`Dockerfile`, `Makefile`, `.env`) and notebooks
are scanned as well. Public URLs and angle-bracket placeholders remain valid,
but a placeholder or `$math$` span is skipped only when it contains no machine
path itself. There is no artificial
"MM-family blocked" import probe: dependency correctness is established by
installing the declared environment and executing the actual paper runtime.

### Gate B — clean-clone runtime

Run from a fresh clone of the exact remote commit, install exactly the
documented requirements, then execute:

```bash
python tools/misc/check_paper.py <name> --runtime
```

The runtime argv comes from the manifest and is executed with `shell=False`.
It must use the current checkout and `sys.executable`; repository discovery
walks up from the script. Do not use machine-specific defaults or silently
delegate to another checkout. Every official config loads; every declared
build config constructs its model; touched core code has focused tests. Run
forward/backward or at least a one-epoch smoke test when the model contract
requires behavior beyond construction. Compare the full test suite against the
merge-base result and permit no new failure.

For an external framework, the runtime log additionally records resolved package
versions, accelerator/compiled-op availability, fallback activation, and the
registry/default scope. The gate must exercise the same config family and
pre/post-processing path that produced the manuscript results. A successful
native-CSRR smoke test cannot stand in for an MMDetection experiment, or vice versa.

### Gate C — experimental evidence and manuscript

This gate is deliberately outside `check_paper.py`. A second reviewer must
recompute seed-level aggregates from the evidence ledger, open every referenced
summary/archive path, audit failed/tuned runs, map headline claims to evidence,
compile the manuscript from clean inputs, inspect the rendered PDF, and verify
the venue’s live requirements. Record each item as `OPEN`, `CONFIRMED`,
`REJECTED WITH EVIDENCE`, `FIXED`, or `RE-VERIFIED`.

Before opening the PR, run all repository gates:

```bash
python tools/misc/check_paper.py <name> --all --base-ref origin/main
```

`REPOSITORY GATES OK; EVIDENCE STILL NOT RUN` is not permission to merge. Merge
only after the external evidence document is independently signed off. After
merge, grep `main` once more and re-run the public commands from a new clone.
