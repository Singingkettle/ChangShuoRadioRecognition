# 新论文代码入库规范

[English](adding_a_new_paper.md) | 简体中文

本文规定一篇新论文的代码如何规范地放进 ChangShuoRadioRecognition (CSRR)。

**算法短名 = `configs/` 下的目录名。** 这是该方法在本仓库的唯一代号。分支名、双 README 两列链接、可选的 `scripts/` 都跟它走。不要再发明第二个名字，也不要改已经进仓库的目录名。

原则：**一篇论文 = `configs/<算法名>/`（配置 + 中英 README 对 + 可选 `scripts/`）+ 需要时的 csrr 原生模块 + 双 README 各一行**。不要发明新的顶层目录，也不要为新论文再建 `tools/<算法名>/` 或 `docs/<算法名>/`。

训练和测试统一从共享入口进入：

```bash
python tools/train.py configs/<算法名>/<config>.py
python tools/test.py configs/<算法名>/<config>.py <checkpoint.pth>
```

参考样例：

- **JDM**：`configs/jdm/README.md` + 根配置 + `configs/jdm/scripts/`。
- **普通 AMC 方法**（如 CNN2）：只有 `configs/cnn2/*.py` + README 对，没有 `scripts/`。
- **DetectionIsEasy**：`tools/detection_is_easy/` + `docs/detection_is_easy/` 是**历史例外，勿仿**。

## 文档语言

所有面向使用者的文档都必须成对。**默认文件是英文。**

- 英文：`foo.md`
- 中文：同目录 `foo_zh-CN.md`
- 文首互链：英文页写 `English | [简体中文](foo_zh-CN.md)`，中文页写 `[English](foo.md) | 简体中文`

新论文必须同时提供 `configs/<算法名>/README.md` 与 `configs/<算法名>/README_zh-CN.md`。不要把方法说明放到 `docs/<算法名>/`。

## 1. configs/<算法名>/ — 配置、说明、可选脚本

- 一篇论文一个文件夹。**文件夹名 = 算法短名**（小写下划线），一经选定不要改。**里面每个配置文件 = 一个实验设置**。
- 命名：`<method>_<modality>-<dataset>.py`。能复现论文数字的变体可放 `experiments/`。
- **`experiments/` 只留复现主线**。围攻失败的变体、manifest、goal 文件不要入库。
- 每个 root 配置头部加 `# Paper:` 注释。
- `_base_` 只允许同目录、`../_base_/`、或 `mmdet::...`。配置路径一律仓库相对路径，禁止机器绝对路径。

### README.md 与 README_zh-CN.md

说明只放 `configs/<算法名>/` 下的 README 对：标题、引用、方法简述、论文章节到代码、数据、训练评测（入口必须是 `tools/train.py` / `tools/test.py`）、结果表、实现偏差。

复现的含义是官方配置跑出来的数字和公开结果别差太远。不要另建 `amr_benchmark` 旁路。不要把围攻日志放进这个文件夹。

### scripts/（可选）

只有该论文真有独特步骤时才建。用向上查找 `tools/train.py` + `csrr/` 定位仓库根。普通分类方法不要建空的 `scripts/`。

## 2. csrr/ — 框架原生模块

新 backbone / head / dataset 放 `csrr/` 并注册。`__init__.py` 是 CRLF。论文专属胶水放 `configs/<算法名>/scripts/`，不要新建 `tools/<算法名>/`。

## 3. tools/ — 共享入口

训练 `tools/train.py`，测试 `tools/test.py`。禁止再为新论文建 `tools/<算法名>/`。DetectionIsEasy 的工具目录是历史例外。

## 4. 顶层 README.md + README_zh-CN.md

Supported Methods 两列都链到 `configs/<算法名>`。两个 README 逐行镜像，CRLF。

## 5. 提交与 PR

分支名 `paper/<算法名>`。contributor 一律是 Singingkettle。author/committer 用 ChangShuo <changshuo@bupt.edu.cn>。禁止 Co-authored-by。提交信息单行、无 conventional 前缀。

## 6. 严禁入库

论文稿件、数据集与重资产、机器绝对路径、密钥、围攻脚本，以及 `amr_benchmark` / `tools/<算法名>/` / `docs/<算法名>/` 旁路（DetectionIsEasy 历史例外除外）。

## 7. 入库前验收清单

```bash
python -m py_compile configs/<算法名>/*.py
grep -rE '/home/' configs/<算法名>
grep -n '<显示名>' README.md README_zh-CN.md
```
