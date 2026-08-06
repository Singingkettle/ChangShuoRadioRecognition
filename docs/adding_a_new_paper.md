# 新论文代码入库规范 (Adding a New Paper's Code)

本文规定一篇新论文的代码如何规范地放进 ChangShuoRadioRecognition (CSRR)。
原则:**一篇论文 = configs 一个文件夹 + tools 一个文件夹 + docs 一个文件夹 + csrr 原生模块 + 双 README 各一行**。
不要发明新的顶层目录(不要建 `projects/` 之类)。参考样例:JDM(`configs/jdm/` +
`tools/jdm/` + `docs/csrd_jointdet/`)与 DetectionIsEasy(`configs/detection_is_easy/` +
`tools/detection_is_easy/` + `docs/detection_is_easy/`)。

## 1. configs/<论文名>/ — 全部实验配置

- 一篇论文一个文件夹,文件夹名 = 论文短代号(小写下划线)。**里面每个配置文件 = 一个实验设置**。
- 命名:`<method>_<modality>-<dataset>.py`(如 `cnn2_iq-deepsig-201610A.py`、`jdm-det_fft-csrd.py`)。
  同一方法的变体可放 `experiments/` 子目录,命名 `<主配置名>_<变体后缀>.py`。
- 每个配置(至少每个 root 配置)头部加注释:

  ```python
  # <一两行方法描述>
  # Paper: "<论文标题>", <期刊/会议> (<年份或 under review>).
  ```

- `_base_` 只允许三种引用:同目录 `./xxx.py`、共享基座 `../_base_/...`、外部包 `mmdet::...`。
  **禁止** `../<别的论文夹>/`、绝对路径、仓库外相对路径。
- 配置内的 `data_root` / `work_dir` / `ann_file` 一律仓库相对路径(`data/...`、`work_dirs/...`)。
  **严禁 `/home/<user>/...` 等机器绝对路径**(投稿代码在别人机器上必须能跑)。

## 2. csrr/ — 框架原生模块

- 新 backbone → `csrr/models/backbones/<name>.py`,`@BACKBONES.register_module()`,继承 `BaseBackbone`,
  `forward` 返回 tuple `(x,)`;新 head → `csrr/models/heads/`,实现 `forward/loss/predict`
  (`loss`/`predict` 消费 `DataSample`);新 dataset → `csrr/datasets/`,`@DATASETS.register_module()`,
  设 `METAINFO = {'classes': (...)}`。
- 在对应 `__init__.py` 加 import 行并把类名加进 `__all__`。**两个陷阱**:
  1. 这些 `__init__.py` 是 **CRLF 行尾** — 用字节级/保行尾的方式编辑,不要整文件重写(会产生
     全文件假 diff);
  2. `__all__` 里前一项若无尾逗号,直接续写会触发 Python 隐式字符串拼接
     (`'a' 'b'` → `'ab'`)— 加新项前确认上一项带逗号。
- 只放"干净可复用"的模型代码;论文专属的胶水/编排逻辑放 tools(见下)。

## 3. tools/<论文名>/ — 论文专属工具(平铺)

- 训练评测 harness、数据生成、桥接/评估、画图脚本全部平铺在这一个文件夹(参考 `tools/jdm/`)。
- 论文如依赖 mmdet 等核心框架不依赖的栈,插件模块(如 `mmdet_plugins.py`)也放这里,并保证:
  - 配置里 `custom_imports = dict(imports=['<模块名>'], allow_failed_imports=False)` 用**裸模块名**;
  - harness 在 `Config.fromfile` 之前把本工具目录插进 `sys.path`
    (`TOOL_DIR = Path(__file__).resolve().parent; sys.path.insert(0, str(TOOL_DIR))`);
  - 目录深度变化时同步改 `repo_root()` 的 `parents[N]`(`tools/<论文名>/x.py` → `parents[2]`)。
- **自包含检查**:逐个确认每条 `import` 在仓库内或 requirements 里能解析 — 从旧实验目录拷脚本时
  最容易带进指向旧工程的死 import(`from <旧包>.xxx import ...`)。
- 额外依赖写 `requirements/<论文名>.txt`(参考 `requirements/detection_is_easy.txt`),
  文件头注释说明用途与安装前提。

## 4. docs/<论文名>/README.md — 论文文档

按 `docs/csrd_jointdet/README.md` / `docs/detection_is_easy/README.md` 的模板写:

1. 标题:`# <代号> — <论文标题>`
2. blockquote 引用(作者、标题、期刊、年份/under review、DOI/arXiv)
3. `## Method in one paragraph`
4. `## Paper section → code map`:`| paper | code |` 两列表,论文章节/公式 → 代码路径
5. `## Data`:数据来源、在盘布局、重资产不入库时给再生成命令
6. `## Train / evaluate`:一个带编号注释的 bash 块,从装依赖到出结果
7. `## Results`:主结果表(带种子数/误差条口径)
8. `## Documented deviations / notes`:与论文的实现偏差、坑、约定

## 5. 顶层 README.md + README_zh-CN.md

- 在 `## Supported Methods` 表加一行(按字母序插入):
  `| [<代号>](configs/<论文名>) | [<论文标题>](docs/<论文名>) |`
- **两个 README 是逐行镜像**:同一行号插入同样内容,两边都改,行号必须一致。
- 两个文件都是 CRLF 行尾,同样注意保行尾编辑。

## 6. 提交与 PR

- 分支名 `paper/<论文短名>`;作者身份用仓库一致的 `ChangShuo <changshuo@bupt.edu.cn>`。
- 提交信息单行、无 conventional-commit 前缀(参考 main 历史);一篇论文的入库尽量整理成
  **一个干净提交**(改进期用 `--amend` + `push --force-with-lease`,PR 被 review 后不再改写历史)。
- PR 保持**纯新增**:除注册行外不动既有代码;diff 里不允许出现与本论文无关的文件。

## 7. 严禁入库

- 论文稿件类文件:`.tex` / `.pdf` / `.bib` / 审稿回复 / 图源 PDF(画图**脚本**可以入,成品图不入)
- 数据集与重资产(memmap、npz 缓存、checkpoint)— 提供再生成脚本,不提交字节
- 机器绝对路径、私人服务器信息、密钥
- 一次性探索脚本(`build_*/aggregate_*/audit_*` 之类的 digest 脚本)— 只入复现主线

## 8. 入库前验收清单

```bash
# 语法门:全部新 .py 可编译
python -m py_compile configs/<论文名>/*.py tools/<论文名>/*.py

# 死引用清零(对新增文件跑,应全部 0 命中)
grep -rE '<旧工程目录名>|/home/|parents\[3\]' configs/<论文名> tools/<论文名>

# __init__ 注册正确(__all__ 无隐式拼接、import 行在位)
python -c "import ast; ast.parse(open('csrr/datasets/__init__.py').read())"

# 双 README 行号一致
grep -n '<代号>' README.md README_zh-CN.md
```

- 服务器实跑:所有配置 `Config.fromfile` 全过;原生模型经 registry 构建 + 前向/反向;
  能训则至少跑 1 个 epoch(离线服务器经 git bundle + scp 同步,缺依赖用本地下载 wheel 离线装)。
- 合并后在 main 上再 grep 复核一遍。
