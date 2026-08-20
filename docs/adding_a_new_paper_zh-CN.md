# 新增一篇论文的代码

[English](adding_a_new_paper.md) | 简体中文

本页说明如何把一篇新论文的代码加入 ChangShuoRadioRecognition(CSRR)。

**算法简称 = `configs/` 下的目录名。** 这是本仓库里该方法的唯一名字。分支名、
两份 README 的两列、以及可选的 `scripts/` 都跟随它。不要另起第二个名字,也不要
重命名已经在树里的目录。

规则:**一篇论文 = `configs/<name>/`(配置 + 中英 README 一对 + 该论文运行所需
的非配置文件)+ 缺模型时补的原生 `csrr/` 模块 + 两份根 README 各加一行。** 不要
新建顶层目录(`projects/` 之类)。不要为新论文加 `tools/<name>/` 或
`docs/<name>/`。

训练与测试一律走共享入口:

```bash
python tools/train.py configs/<name>/<config>.py
python tools/test.py configs/<name>/<config>.py <checkpoint.pth>
```

示例:

- **JDM**:`configs/jdm/README.md` + 根配置 + `configs/jdm/scripts/`(合并权重、
  渲染图、预算 proposals 等论文专属步骤)。
- **典型 AMC 方法**(CNN2):只有 `configs/cnn2/*.py` + README 一对。
- **DetectionIsEasy**:唯一一个**有文档记录的依赖例外**——它的检测段用 `mmdet`。
  该依赖**只**隔离在 `requirements/detection_is_easy.txt`,**永远不是核心依赖**。
  新论文不要模仿它;新论文一律原生实现(见 §0)。

## 0. 依赖与框架边界(先读)

CSRR **只**建立在 `mmengine` 之上,这是刻意为之。

- **`csrr/` 核心必须只靠 `mmengine` 就能 import 并运行。** 任何 `csrr/`
  模块不得硬 import 其他 MM 家族包(`import mmcv`、`import mmdet`……)。
  MM 家族原本提供的图像小工具一律改用 `cv2` / `PIL` / `numpy` 实现。

- **核心安装只依赖 `mmengine`(及 PyTorch)。不要再引入其他 MM 家族库**——
  `mmdet`、`mmcv`、`mmpretrain`、`mmsegmentation`……。MM 家族体量大、耦合紧、
  版本脆弱,引入它会让环境臃肿、跨机器复现困难。
- **核心依赖版本钉死、保持不变。** 不要为一篇新论文抬版本。若某论文确需更新的
  核心,单独提出。
- **本仓库没有的模型,在 `csrr/` 下原生实现**(§2),注册进 CSRR registry。
  **不要**为拿一个 backbone / head / detector 而去引外部 MM 家族包。原生模型
  文件放 `csrr/models/…`,并**与论文的 scripts 分开**。
- **一篇论文运行所需的一切,除它的配置文件和原生模型文件外,都放在
  `configs/<name>/`。** 不要把论文的任何东西放到 `tools/<name>/`、`projects/`
  或 `docs/<name>/`。
- 论文专属的额外依赖可以钉在 `requirements/<name>.txt`,但**强烈不建议在这里加
  MM 家族库**;它只留给 DetectionIsEasy 的 mmdet 例外——该例外保持隔离且可选。
- **钉精确版本,不用范围。** 核心钉 `mmengine==<版本>`;论文额外依赖同样
  `pkg==x.y.z`。范围钉(`>=a,<b`)跨机器漂移,破坏字节级复现。
- requirements 地图:`requirements/runtime.txt` = `setup.py` 实际安装的
  (**必须保持无 MM 家族**);`requirements/<name>.txt` = 单篇论文的隔离额外
  依赖;`mminstall.txt` 属遗留,不得再增长。

仓库地图——每个顶层目录只有一个职责:

| 目录 | 职责 |
|---|---|
| `csrr/` | 框架本体:模型、数据集、transform、metric、engine(§2) |
| `csrr/performance/` | 论文图表 / 结果汇总模块(图、指标表) |
| `csrr/apis/` | 推理入口 |
| `configs/_base_/` | 跨论文共享的基配置(datasets、schedules、runtimes) |
| `configs/<name>/` | 一篇论文:配置 + README 对 + 其运行文件(§1) |
| `tools/` | 只放共享的 train/test 入口(§3) |
| `docs/` | 只放框架级文档(安装、上手)+ `docs/dataset/` 共享数据集说明。**不放论文文档。** |
| `tests/` | 单元测试;在 `csrr/evaluation/` 加新 metric 应配 `tests/test_evaluation/` 测试 |
| `requirements/` | 依赖钉版,按上面的地图 |

## 文档语言

每一份面向用户的文档都是一对。**英文是默认文件。**

- 英文:`foo.md`
- 中文:同目录下 `foo_zh-CN.md`
- 头部链接:英文页 `English | [简体中文](foo_zh-CN.md)`,中文页
  `[English](foo.md) | 简体中文`

新论文必须提供 `configs/<name>/README.md` 与
`configs/<name>/README_zh-CN.md`。不要把方法说明放到 `docs/<name>/`。

## 1. configs/<name>/ — 配置、说明、该论文的运行文件

- 一篇论文一个文件夹。**文件夹名 = 简称**(小写、下划线),之后不改。
  **每个配置文件 = 一个实验。**
- 命名:`<method>_<modality>-<dataset>.py`(如 `cnn2_iq-deepsig-201610A.py`、
  `jdm-det_fft-csrd.py`)。真正复现论文数字的变体可放 `experiments/`,命名
  `<main-config>_<suffix>.py`。
- **`experiments/` 只放复现主线**(论文协议评测 + 你实际汇报的工作点)。失败的
  lr / epoch / seed / EMA / SWA 变体、清单、目标文件都不入 git。
- 每个根配置在靠近开头处带上这段——前面可以有 Apache 许可块;保持在前几行
  注释以内:

  ```python
  # <一两行方法说明>
  # Paper: "<title>", <venue> (<year or under review>).
  ```

- `_base_` 可以指向:(a) 同目录 `./xxx.py`;(b) **本论文自己 `configs/<name>/`
  内**的另一个配置——例如 `configs/<name>/experiments/` 下的变体通过
  `../<root-config>.py` 继承论文根配置(JDM 模板正是这么做的);(c) 共享
  `../_base_/...`;或 (d) 外部 `mmdet::...`**(mmdet 仅限 DetectionIsEasy
  例外)**。**不允许** `../<other-paper>/`、绝对路径、或仓库外路径。每个
  `_base_` 目标都必须能在盘上解析到。
- `data_root` / `work_dir` / `ann_file` 用仓库相对路径(`data/...`、
  `work_dirs/...`)。**不允许** `/home/<user>/...` 机器路径。

### configs/<name>/ 下的运行文件

论文运行所需的非配置文件——绘图脚本、数据准备脚本、mmdet 插件模块、评测辅助——
都放在 `configs/<name>/`。保持扁平,不要再嵌 `tools/`。原生模型代码**不**放这里,
放 `csrr/`(§2)。

- 脚本用仓库相对路径。**向上走到同时存在 `tools/train.py` 和 `csrr/` 处作为
  仓库根来定位。不要写死 `parents[N]` 或 `/home/<user>/...`。**
- 每个 `import` 必须能在仓库内或 requirements 里解析。
- 对 DetectionIsEasy 的 mmdet 例外,插件模块也放这里:配置用裸模块名
  `custom_imports = dict(imports=['<module>'], allow_failed_imports=False)`,
  调用方在 `Config.fromfile` 前把本目录插入 `sys.path`。
- 额外依赖写 `requirements/<name>.txt`(参考 `requirements/detection_is_easy.txt`),
  带一段用途与安装前提的头注释。MM 家族额外依赖不建议(§0)。

### README.md 与 README_zh-CN.md

说明只放 `configs/<name>/` 下的 README 一对。用这个大纲:

1. 标题:`# <展示名> — <论文标题>`
2. 引用块(作者、标题、venue、年份 / under review、DOI / arXiv)
3. `## Method in one paragraph`
4. `## Paper section → code map`:`| paper | code |`
5. `## Data`:数据来源、盘上布局、重资产不入 git 时的再生成命令;记录任何与公开
   协议不同的划分(此处 DeepSig 用 50/10/40)
6. `## Train / evaluate`:一段编号 bash。入口必须是 `tools/train.py` /
   `tools/test.py`,论文专属步骤用 `python configs/<name>/...`
7. `## Results`:实测 vs 已发表数字,含种子 / 误差棒政策
8. `## Documented deviations / notes`

复现的含义:官方 `configs/<name>/` 跑出来不应离已发表数字太远。不要造一个
`amr_benchmark`(或类似)旁路文件夹去代表"一组算法"。

不要提交围攻日志(`retune_campaign.md`、`retune_results.md`、`goal_mode.md`)。
主表和偏差写进 README。

## 2. csrr/ — 原生框架模块(所有缺失的组件都放这里)

若论文需要一个 CSRR 没有的**框架组件**——不只是模型,还包括数据集、特殊的数据
加载/预处理 transform、sampler、filter、评测 metric、loss——**在对应的 `csrr/`
子包里原生实现并注册。** 不要为它引 MM 家族包(§0),也**不要一股脑塞进
`configs/<name>/` 和配置文件混在一起。** CSRR 有完整的 registry 体系(见
`csrr/registry.py`:`DATASETS`、`TRANSFORMS`、`METRICS`、`MODELS`、
`DATA_SAMPLERS`、`DATA_FILTERS`……),每类可复用组件都有归属。

各类组件的归属(CSRR 子包 → registry):

| 要加的组件 | 放到 | 注册方式 |
|---|---|---|
| 模型 backbone | `csrr/models/backbones/` | `@MODELS`——继承 `BaseBackbone`,`forward` 返回 `(x,)` |
| detector / classifier | `csrr/models/{detectors,classifiers}/` | `@MODELS` |
| head(在 `DataSample` 上实现 `forward` / `loss` / `predict`) | `csrr/models/heads/` | `@MODELS` |
| loss | `csrr/models/losses/` | `@MODELS` |
| 数据集 | `csrr/datasets/` | `@DATASETS`,设 `METAINFO = {'classes': (...)}` |
| **数据加载 / 预处理 transform** | **`csrr/datasets/transforms/`** | `@TRANSFORMS` |
| sampler / filter | `csrr/datasets/{samplers,filters}/` | `@DATA_SAMPLERS` / `@DATA_FILTERS` |
| 评测 metric / evaluator | `csrr/evaluation/metrics/` | `@METRICS` |
| hook / optimizer / scheduler | `csrr/engine/` | `@HOOKS` / `@OPTIMIZERS` / `@PARAM_SCHEDULERS` |
| 数据结构 | `csrr/structures/` | — |
| 可视化器 | `csrr/visualization/` | `@VISUALIZERS` |

`MODELS` 是所有模型部件的唯一归属:`csrr/models/builder.py` 把 `BACKBONES`、
`NECKS`、`HEADS`、`LOSSES`、`CLASSIFIERS` 都别名到它,所以旧代码的
`@BACKBONES.register_module()` 与新写的 `@MODELS.register_module()` 落进同一个
registry——用 `@MODELS` 注册即可。想让论文图表可复用,可注册进
`csrr/performance/`(`@FIGURES` / `@TABLES`,由 `tools/analyze.py` 驱动),就像
JDM 发布它的图那样;单篇论文用 `configs/<name>/` 下一个普通绘图脚本也可以。

不要照搬空脚手架。有些声明了名字却没有实现——指向不存在的 `csrr.analysis`
模块的 `ANALYSIS` registry、`csrr/apis/` 里近乎空的角落。把组件放进真正接线了的
registry 所在子包(上表),并确认 `register_all_modules()` 能 import 到它。

- **自定义的数据集加载方式是框架组件**:归 `csrr/datasets/transforms/`
  (`@TRANSFORMS`),不是塞进 `configs/<name>/`。自定义 metric、sampler、loss
  同理——各回各的 `csrr/` 子包,不和配置文件混放。
- 这些文件留在 `csrr/`,**与论文的 `configs/<name>/` 运行文件分开。**
- 把 import 和类名加进对应 `__init__.py`。两个坑:
  1. 这些 `__init__.py` 是 **CRLF**。按字节编辑,不要整文件重写(会产生假的
     全文件 diff)。
  2. 若 `__all__` 最后一项没有尾逗号,追加名字会静默拼接字符串
     (`'a' 'b'` → `'ab'`)。确认上一项带逗号。
- 这里只放可复用的框架代码。**论文专属胶水**——绘图脚本、数据生成编排、以及
  (DetectionIsEasy 例外的)mmdet 插件模块——放 `configs/<name>/`,不放
  `tools/<name>/`。mmdet 例外的插件是唯一放 `configs/<name>/` 而非 `csrr/`
  的情况,因为它注册进的是 mmdet 而不是 CSRR 原生 registry。

## 3. tools/ — 共享入口,不是每篇论文的文件夹

- **训练:** `tools/train.py`
- **测试:** `tools/test.py`(分类把 `pred_score` 收进 `paper.pkl`;检测 / 联合
  配置用 mmengine 的 `Runner.test()`)
- 共享零件可留在 `tools/analyze.py`、`tools/convert_datasets/`、`tools/misc/`
- **不要**为新论文加 `tools/<name>/`。论文的运行文件放 `configs/<name>/`(§1),
  模型放 `csrr/`(§2)。

## 4. 根 README.md + README_zh-CN.md

- 在 `## Supported Methods` 下按字母序加一行:
  `| [<display>](configs/<name>) | [<paper title>](configs/<name>) |`
- 展示名可用论文惯用写法(`JDM`),但**两列都链到 `configs/<name>`**。
- 两份 README 逐行镜像:同一插入行号、两个文件、同行号。
- 两份都是 CRLF,保留行尾。

## 5. 提交与 PR

- 分支名 `paper/<name>`,与 `configs/` 目录一致。
- **唯一贡献者是 [Singingkettle](https://github.com/Singingkettle)。** git
  author/committer 为该账号的 `ChangShuo <changshuo@bupt.edu.cn>`。
  **不要加 `Co-authored-by:` 行。** 不要把 Cursor、助手、或其他 GitHub 账号
  放进贡献者、PR 作者或提交元数据。
- 单行提交信息,无 conventional-commit 前缀(跟 `main`)。每篇论文尽量
  **一个干净提交**(迭代期用 `--amend` + `--force-with-lease`;评审后不改历史)。
- **push 前先拉取最新 `main`** 并 rebase 其上。PR 保持**纯 add**:不动无关代码。
  diff 里不得含不属于本论文的文件。

## 6. 不要提交

- **核心** requirements 里出现另一个 MM 家族库(`mmdet`、`mmcv`、`mmpretrain`……)。
  核心保持只 `mmengine`(§0);DetectionIsEasy 的 mmdet 例外隔离在
  `requirements/detection_is_easy.txt`。
- 稿件文件:`.tex` / `.pdf` / `.bib` / 审稿回复 / 图源 PDF(绘图**脚本**可以,
  成品图不行)
- 数据集与重资产(memmap、npz 缓存、checkpoint、预测转储、`.jsonl` 诊断)——
  提供再生成脚本,不是字节
- 机器绝对路径、私有服务器细节、密钥
- 一次性探索(`build_*` / `aggregate_*` / `audit_*`、keepalive、sweep 编排)与
  失败的围攻配置——只留在主线本地
- `amr_benchmark`、`tools/<name>/`、`projects/`、`docs/<name>/` 旁路

## 7. 合并前检查清单

先跑验收脚本。它把下面每一项都机器化了,开 PR 前必须打印 `RESULT: OK`:

```bash
python tools/misc/check_paper.py <name>
```

它在 HARD 规则上失败(`RESULT: FAILED`,非零退出)——README 一对含中文镜像 +
头部切换链接、至少一个 `# Paper:` 头、`_base_` 合法且能在盘上解析、无 `/home/`
或写死 `parents[N]` 路径、`py_compile`、两份根 README 同行号且**两列**都链
`configs/<name>`、屏蔽 `mmcv` 后 `import csrr` 成功、`requirements/<name>.txt`
钉精确 `==`——只在软规则上 WARN(文件名卫生、逐配置 `# Paper:` 覆盖)。它只依赖
标准库、只用 `mmengine` 即可运行。下面这些命令是它所自动化的内容,便于手动核对
单条规则:

```bash
# 语法:每个新 .py 能编译
python -m py_compile configs/<name>/*.py
# 若论文在 configs/<name>/ 下带运行文件:
python -m py_compile configs/<name>/**/*.py

# 死引用(新文件必须零命中)
grep -rE '<old-project-name>|/home/|parents\[' configs/<name>

# registry __init__ 有效(无隐式字符串拼接;import 存在)
python -c "import ast; ast.parse(open('csrr/datasets/__init__.py').read())"

# 核心 import 只靠 mmengine 就能通过(mmcv/mmdet 缺席或被屏蔽)
python -c "import csrr"   # 在只有 mmengine 的环境里
# 核心保持只 mmengine:核心 requirements 里没有 MM 家族
grep -riE 'mmcv|mmdet|mmpretrain' requirements/*.txt   # 只允许 detection_is_easy.txt 命中

# 每份 .md 都有中文对偶 + 头部切换链接
ls configs/<name>/*_zh-CN.md

# 两份根 README 同行号,两列都链到 configs/<name>
grep -n '<display>' README.md README_zh-CN.md
```

- 服务器上:每个配置能 `Config.fromfile`;原生模型能从 registry 构建并前向 /
  反向;能训练就至少跑一个 epoch。
- 合并后再 grep 一次 `main`。
