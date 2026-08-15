# 新论文代码入库规范 (Adding a New Paper's Code)

本文规定一篇新论文的代码如何规范地放进 ChangShuoRadioRecognition (CSRR)。

**算法短名 = `configs/` 下的目录名。** 这是该方法在本仓库的唯一代号。分支名、
README 两列链接、可选的 `scripts/` 都跟它走，不要再发明第二个名字，也不要改
已经进仓库的目录名。

原则:**一篇论文 = `configs/<算法名>/`（配置 + `README.md` + 可选 `scripts/`）
+ 需要时的 csrr 原生模块 + 双 README 各一行**。不要发明新的顶层目录（不要建
`projects/` 之类），也不要为新论文再建 `tools/<算法名>/` 或 `docs/<算法名>/`。

训练和测试统一从共享入口进入：

```bash
python tools/train.py configs/<算法名>/<config>.py
python tools/test.py configs/<算法名>/<config>.py <checkpoint.pth>
```

参考样例:

- **JDM**:`configs/jdm/README.md` + 根配置 + `configs/jdm/scripts/`（合并权重、
  提图、proposal 预计算等该论文独有的步骤）。
- **普通 AMC 方法**（如 CNN2）:只有 `configs/cnn2/*.py` + `configs/cnn2/README.md`，
  没有 `scripts/`。
- **DetectionIsEasy**:`tools/detection_is_easy/` + `docs/detection_is_easy/` 是
  **历史例外，勿仿**。新论文不要再按这个三件套做。

## 1. configs/<算法名>/ — 配置、说明、可选脚本

- 一篇论文一个文件夹。**文件夹名 = 算法短名**（小写下划线），一经选定不要改。
  **里面每个配置文件 = 一个实验设置**。
- 命名:`<method>_<modality>-<dataset>.py`（如 `cnn2_iq-deepsig-201610A.py`、
  `jdm-det_fft-csrd.py`）。同一方法、能复现论文数字的变体可放 `experiments/`
  子目录，命名 `<主配置名>_<变体后缀>.py`。
- **`experiments/` 只留复现主线**（论文协议评测、实际采用的工作点）。围攻失败
  的 lr/epoch/seed/EMA/SWA 变体、manifest、goal 文件不要入库。
- 每个配置（至少每个 root 配置）头部加注释:

  ```python
  # <一两行方法描述>
  # Paper: "<论文标题>", <期刊/会议> (<年份或 under review>).
  ```

- `_base_` 只允许三种引用:同目录 `./xxx.py`、共享基座 `../_base_/...`、外部包
  `mmdet::...`。**禁止** `../<别的论文夹>/`、绝对路径、仓库外相对路径。
- 配置内的 `data_root` / `work_dir` / `ann_file` 一律仓库相对路径（`data/...`、
  `work_dirs/...`）。**严禁 `/home/<user>/...` 等机器绝对路径**（投稿代码在别人
  机器上必须能跑）。

### README.md

说明文档只放 `configs/<算法名>/README.md`，按这个模板写:

1. 标题:`# <显示名> — <论文标题>`
2. blockquote 引用（作者、标题、期刊、年份/under review、DOI/arXiv）
3. `## Method in one paragraph`
4. `## Paper section → code map`:`| paper | code |` 两列表，论文章节/公式 → 代码路径
5. `## Data`:数据来源、在盘布局、重资产不入库时给再生成命令；写清和公开协议
   不一致的划分（例如 DeepSig 用 50/10/40）
6. `## Train / evaluate`:一个带编号注释的 bash 块，入口必须是 `tools/train.py`
   / `tools/test.py`；该论文独有的步骤写 `python configs/<算法名>/scripts/...`
7. `## Results`:主结果表（测得 vs 公开数字，带种子数/误差条口径）
8. `## Documented deviations / notes`:与论文的实现偏差、坑、约定

复现的含义是：官方 `configs/<算法名>/` 跑出来的数字和公开结果别差太远。不要
另建 `amr_benchmark` 这类旁路目录来代表「一组算法」。

不要把围攻日志（`retune_campaign.md` / `retune_results.md` / `goal_mode.md`
之类）放进这个文件夹。主结果和偏差写进 README 即可。

### scripts/（可选）

只有该论文真有独特步骤时才建 `configs/<算法名>/scripts/`，例如合并两个模块的
checkpoint、生成论文图、预计算 proposal。目录平铺，不要再套一层 `tools/`。

- 脚本用仓库相对路径；用向上查找 `tools/train.py` + `csrr/` 的方式定位仓库根，
  不要写死 `parents[N]`，也不要写 `/home/<user>/...`。
- 论文如依赖 mmdet 等核心框架不依赖的栈，插件模块也可放这里，并保证:
  - 配置里 `custom_imports = dict(imports=['<模块名>'], allow_failed_imports=False)`
    用**裸模块名**;
  - 调用方在 `Config.fromfile` 之前把 `scripts/` 插进 `sys.path`。
- **自包含检查**:逐个确认每条 `import` 在仓库内或 requirements 里能解析。
- 额外依赖写 `requirements/<算法名>.txt`（参考 `requirements/detection_is_easy.txt`），
  文件头注释说明用途与安装前提。

普通分类方法共用 `tools/train.py` / `tools/test.py`，**不要**为它们建空的
`scripts/`。

## 2. csrr/ — 框架原生模块

- 新 backbone → `csrr/models/backbones/<name>.py`，`@BACKBONES.register_module()`，
  继承 `BaseBackbone`，`forward` 返回 tuple `(x,)`；新 head → `csrr/models/heads/`，
  实现 `forward/loss/predict`（`loss`/`predict` 消费 `DataSample`）；新 dataset →
  `csrr/datasets/`，`@DATASETS.register_module()`，设 `METAINFO = {'classes': (...)}`。
- 在对应 `__init__.py` 加 import 行并把类名加进 `__all__`。**两个陷阱**:
  1. 这些 `__init__.py` 是 **CRLF 行尾** — 用字节级/保行尾的方式编辑，不要整文件
     重写（会产生全文件假 diff）；
  2. `__all__` 里前一项若无尾逗号，直接续写会触发 Python 隐式字符串拼接
     (`'a' 'b'` → `'ab'`)— 加新项前确认上一项带逗号。
- 只放「干净可复用」的模型代码；论文专属的胶水/编排逻辑放
  `configs/<算法名>/scripts/`，不要新建 `tools/<算法名>/`。

## 3. tools/ — 共享入口，不是论文目录

- **训练**:`tools/train.py`
- **测试**:`tools/test.py`（分类收 `pred_score` 写 `paper.pkl`；检测 / joint
  配置走 mmengine `Runner.test()`）
- 共享杂项可以留在 `tools/analyze.py`、`tools/convert_datasets/`、`tools/misc/`
- **禁止**再为新论文建 `tools/<算法名>/`
- DetectionIsEasy 的 `tools/detection_is_easy/` 是历史例外，勿仿

## 4. 顶层 README.md + README_zh-CN.md

- 在 `## Supported Methods` 表加一行（按字母序插入）:
  `| [<显示名>](configs/<算法名>) | [<论文标题>](configs/<算法名>) |`
- 显示名可以是论文常用写法（如 `JDM`），但 **两列链接都链到 `configs/<算法名>`**。
- **两个 README 是逐行镜像**:同一行号插入同样内容，两边都改，行号必须一致。
- 两个文件都是 CRLF 行尾，同样注意保行尾编辑。

## 5. 提交与 PR

- 分支名 `paper/<算法名>`，与 `configs/` 目录名一致。
- **contributor 一律是 [Singingkettle](https://github.com/Singingkettle)**。git
  author/committer 用该账号绑定的 `ChangShuo <changshuo@bupt.edu.cn>`。
  **禁止** `Co-authored-by:` 行，禁止把 Cursor / 助手 / 其他 GitHub 账号写进
  contributors、PR 作者或提交元数据。
- 提交信息单行、无 conventional-commit 前缀（参考 main 历史）；一篇论文的入库尽量
  整理成 **一个干净提交**（改进期用 `--amend` + `push --force-with-lease`，PR 被
  review 后不再改写历史）。
- PR 保持**纯新增**:除注册行外不动既有代码；diff 里不允许出现与本论文无关的文件。

## 6. 严禁入库

- 论文稿件类文件:`.tex` / `.pdf` / `.bib` / 审稿回复 / 图源 PDF（画图**脚本**可以
  入，成品图不入）
- 数据集与重资产（memmap、npz 缓存、checkpoint）— 提供再生成脚本，不提交字节
- 机器绝对路径、私人服务器信息、密钥
- 一次性探索脚本（`build_*/aggregate_*/audit_*`、keepalive、sweep orchestrator）
  和围攻失败配置 — 只入复现主线
- `amr_benchmark`、`tools/<算法名>/`、`docs/<算法名>/` 这类旁路（DetectionIsEasy
  历史例外除外）

## 7. 入库前验收清单

```bash
# 语法门:全部新 .py 可编译
python -m py_compile configs/<算法名>/*.py
# 若有专属脚本:
python -m py_compile configs/<算法名>/scripts/*.py

# 死引用清零(对新增文件跑,应全部 0 命中)
grep -rE '<旧工程目录名>|/home/' configs/<算法名>

# __init__ 注册正确(__all__ 无隐式拼接、import 行在位)
python -c "import ast; ast.parse(open('csrr/datasets/__init__.py').read())"

# 双 README 行号一致;两列都链到 configs/<算法名>
grep -n '<显示名>' README.md README_zh-CN.md
```

- 服务器实跑:所有配置 `Config.fromfile` 全过；原生模型经 registry 构建 + 前向/
  反向；能训则至少跑 1 个 epoch（离线服务器经 git bundle + scp 同步，缺依赖用本地
  下载 wheel 离线装）。
- 合并后在 main 上再 grep 复核一遍。
