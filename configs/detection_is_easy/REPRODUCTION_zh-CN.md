# 复现记录

[English](REPRODUCTION.md) | 简体中文

本文件记录从一份干净克隆的已发布代码端到端重跑论文实验，并逐格说明重跑相对
报告数字落在哪里。这是事实日志。它不下测量本身逼不出来的结论。

下面每个数字都在服务器硬件上、从已发布的 `main` 分支产出。对比表由
`configs/detection_is_easy/collect_repro_results.py` 对照
`configs/detection_is_easy/paper_values.csv` 机械生成，该文件把每个报告值存在
它所属的表或章节旁边。

## 跑了什么，在什么上面跑

- **代码**：对本仓库做干净 `git clone`，commit `88c02ff`（下面两处数据路径
  修复已在其中）。报告工具 `collect_repro_results.py` 之后加在 `c997b1c`；
  它只读结果，不改结果。
- **环境**：严格按 `requirements/detection_is_easy.txt` 新建的 venv —
  torch 2.7.1+cu128，numpy 2.2.6，mmdet 3.3.0，**mmcv-lite 2.1.0**（无编译
  `_ext`），mmengine 0.10.7，torchsig 2.1.1。已按包对照论文运行环境核验。
- **硬件**：8x RTX 4090。
- **数据**：已发布的 `hardshort_lowsnr` 基准 — 50000/5000/10000 训练/验证/测试，
  57 类 — 57 类标注在本机用 `build_multiclass_coco.py` 重建（匹配率 1.0000，
  框数与原版相同）。

## 一格怎样算复现

`cfg.randomness = dict(seed=..., deterministic=False)`，所以 cuDNN 会选非确定性
核，同一种子的两次运行会不同。我们在评判之前先测了这个地板：RTMDet-M 三次
相同运行的 sd 为 **0.0076**，RTMDet-tiny 为 **0.0033**。当种子均值落在报告值的
**0.023**（较大地板的三倍）以内时，该格算复现。这是一条带，不是一个点；
本基准上单对运行撑不起 0.01 的差异。

两个运行时契约很重要，并按次记在 `run_info.json`：

- **`used_mmcv_lite_stub: true`** — 每个结果都用纯 PyTorch NMS 回退，因为论文
  环境没有编译过的 mmcv。装完整 CUDA mmcv 会换 NMS 实现，数字会略偏。
- **`used_pytorch_focal_loss: true`** — FCOS 与 ATSS 需要 mmcv-lite 不带的
  focal-loss 核；工具把它们转到 mmdet 自己的 `py_sigmoid_focal_loss`（同一个量）。
  见下面的 “已修复的可复现性缺陷”。

## 检测器格子：实测对报告

57 类上的类感知 `coco/bbox_mAP`，验证划分。`n` 是种子数；`sd` 是跨种子散布。
“verdict” 对照 0.023 带。

| Cell | reproduced | sd | paper | delta | verdict | paper source |
|---|---|---|---|---|---|---|
| Axis B, tiny, uniform | 0.432 | 0.0035 | 0.431 | +0.001 | within | Table I uniform |
| Axis B, small, uniform | 0.443 | 0.0205 | 0.449 | -0.006 | within | Table I uniform |
| Axis B, medium, uniform | 0.472 | 0.0121 | 0.460 | +0.012 | within | Table I uniform |
| Axis B, large, uniform | 0.451 | 0.0074 | 0.462 | -0.011 | within | Table I uniform |
| Axis B, tiny, own sched. | 0.433 | 0.0035 | 0.408 | +0.025 | **outside** | Table I own |
| Axis B, small, own sched. | 0.470 | 0.0135 | 0.429 | +0.041 | **outside** | Table I own |
| Axis B, medium, own sched. | 0.486 | 0.0152 | 0.477 | +0.009 | within | Table I own |
| Axis B, large, own sched. | 0.504 | 0.0154 | 0.486 | +0.018 | within | Table I own |
| Axis A, magnitude-only | 0.441 | 0.0231 | 0.455 | -0.014 | within | Table III |
| Axis A, phase+magnitude | 0.455 | 0.0125 | 0.447 | +0.008 | within | Table III |
| Axis A, phase only | 0.431 | -- | 0.440 | -0.009 | within | Table III |
| Axis A, learnable filterbank | 0.418 | -- | 0.412 | +0.006 | within | Table III |
| Axis E, complex-1D + FFT | 0.053 | -- | 0.026 | +0.027 | **outside** | Table III |
| FCOS | 0.470 | 0.0053 | 0.374 | +0.096 | **outside** | Section VI-B |
| ATSS | 0.468 | 0.0032 | 0.380 | +0.088 | **outside** | Section VI-B |
| localization (single-class) | 0.893 | 0.0060 | 0.948 | -0.055 | **outside** | Sections I, IV, VI-A |
| deployment detector (best run) | 0.472 | -- | 0.521 | -0.049 | **outside** | Table I caption / VI-D |

没有自己论文值的对照：

| Cell | reproduced | sd | why it was run |
|---|---|---|---|
| RTMDet-M, uniform, batch 4 | 0.426 | 0.0006 | batch-matched family control (§VI-B) |
| RTMDet-M, own sched., batch 4 | 0.491 | -- | batch-decoupled size control (§VI-C, L5) |

带论文值的 21 格里有 14 格落在带内。其余 7 格各自在下面交代。

## 识别器与部署格子

识别器在验证裁剪缓存上的准确率（干净信号），以及部署桥在 2963 个测试场景上、
用归档原始 IQ、从复现的部署检测器出发。

| Cell | metric | reproduced | paper | note |
|---|---|---|---|---|
| recipe-A recognizer (3 seeds) | combined fine acc | 0.875 / 0.876 / 0.871 | 0.869 | run-index 19 |
| 40-epoch predecessor | stage2-single acc | 0.632 | 0.643 | run-index 20 -- see dose-response note |
| 40-epoch predecessor | combined fine acc | 0.534 | -- | |
| recipe-B (mixup) | combined fine acc | 0.699 | 0.714 | run-index 21 |
| differential phase | combined fine acc | 0.913 | 0.916 | run-index 26 |
| recipe-A deployment (3 seeds) | fused delta | +0.028 (0.027/0.029/0.027) | +0.024 | run-index 22 |
| recipe-A deployment | psk / ask / qam delta | +0.153 / +0.132 / +0.081 | +0.143 / +0.118 / +0.084 | run-index 23 |
| oracle (perfect box), recipe-A | pure-IQ class mAP | 0.608 | 0.608 | run-index 24 |
| differential phase deployment | fused delta | +0.022 | +0.019 | run-index 26 |

部署桥使用记录下来的非默认开关 — `--score-thr 0.05 --limit 2963
--class-nms-iou 0.5 --ours-score-recog --iq-families psk,ask,qam`。用桥的默认值
跑会评另一组场景、关掉逐类 NMS，复现不了这些数字。

## 七个带外格子

**FCOS +0.096，ATSS +0.088（§VI-B）。** 报告的 FCOS/ATSS 运行用恒等归一化训练 —
生成配置带着 `mean=[0,0,0] std=[1,1,1]`，而数据的逐通道 sigma 大约 12.8 —
它们对照的 RTMDet 运行却用了真实统计。用注入统计重跑 FCOS/ATSS
（`--require-tensor-stats`）得到 0.470 与 0.468，三种子 sd 为 0.005 与 0.003。
另外两件事把图景收得更紧：

- 同一 uniform 配方下的 RTMDet-M 在 batch 8 是 0.472，但在 FCOS/ATSS 所用的
  batch 4 上是 **0.426**。匹配 batch 时，FCOS（0.470）与 ATSS（0.468）*高于*
  RTMDet-M（0.426），而不是低于它。
- 所以报告的 RTMDet 领先来自两个叠加伪影：归一化缺陷拖累其他头，以及 batch
  大小不同。两者都去掉后，检测器家族分不开。这与论文自己的论点一致 — 定位
  已饱和，缺口在识别 — 但它翻转了 §VI-B 那句“头与分配器重要、RTMDet 是特别
  有理由的选择”。

**定位 −0.055（§I/IV/VI-A）。** 报告的类无关定位 mAP 0.948 测在比已发布
`hardshort_lowsnr` 基准更容易的生成器配置上（`lowsnr` 集：更长时长、更高 SNR、
无同信道重叠）。在已发布基准上三次单类运行给出 **0.893 +- 0.006**。定位仍是
容易、接近饱和的轴；陈述它的数字应当是在论文分发的基准上测得的那个。

**部署检测器 −0.049。** 论文把这一格的单次最好运行（0.521）当作部署基线，却把
三种子均值（0.477）用于尺寸扫描。复现给出单次 0.472 — 三种子均值 0.486，
落在带内。部署数字都从该 checkpoint 桥过去，所以绝对水平跟着它走
（复现基线 0.474 对论文的 0.522）；报告的*增量*无论怎样都能复现。

**small/own +0.041，tiny/own +0.025。** own-schedule 列上的真实偏差，该列报告的
散布本身就大（论文自己的三种子 sd 为 0.017 与 0.041）。同样尺寸的 uniform 列
干净复现。

**complex-1D +0.027。** 两个值都是塌缩（0.053 vs 0.026）；在那个量级上差值没有
意义。Axis E 的发现 — 先学习再 FFT 的前端会毁掉定位 — 可以复现。

## 复现挖出的两个发现

**预测框识别器赢 +0.093，而不是 −0.019（§VII）。** “在预测框上训识别器”的负表
行报告部署增量为 −0.019。复现给出 **+0.093**，三种子
（+0.0930/+0.0922/+0.0919，sd < 0.001），对照 recipe-A 基线 +0.028。三处检查
钉住机制：

- *不是泄漏。* 训练与测试场景不相交（50000 对 10000，没有共享样本 id 或文件名）。
- *是分布匹配，不是更好的识别器。* 在完美 GT 框上该识别器得 0.415，*低于*
  recipe-A 的 0.608，干净验证准确率是 0.525 对 recipe-A 的 0.875。除了它为之
  训练的预测框分布，它处处更差。
- *解释了报告的 −0.019。* 论文的裁剪缓存名叫 `trainpred_hi`；构建参数从未记录，
  里面有 194k 个裁剪。用高检测分数截断（`--score-thr 0.5`）建类似缓存，留下
  87k 个接近完美的框，部署增量是 **−0.027** — 复现了论文的 −0.019。在完整
  路由框分布上训练（`--score-thr 0.1`）才给出 +0.093。发表的负结果是从高分、
  接近 GT 的框建缓存造成的伪影。

`trainpred_hi` 的参数没有记录；+0.093 缓存按已发布默认 `--score-thr 0.1` 在全部
50000 场景上构建，再按固定种子随机子采样到与 GT 缓存相同的裁剪数，以免比较
被多出来的数据带走。

**剂量响应曲线混用了两种指标（§VI-D）。** 报告的 0.643 -> 0.714 -> 0.869
从三次运行各取了一种不同指标。干净地测，每种指标各自都是单调的：
stage2-single 准确率 0.632 / 0.691 / 0.867，combined fine 准确率 0.534 / 0.699 /
0.875。报告的 0.643 是被标成 combined 的 stage2-single 数字。

## 已修复的可复现性缺陷（在已发布代码里）

从干净克隆跑的过程中发现；每一处都已提交到 `main`：

- **FCOS/ATSS 在 mmcv-lite 下无法训练**（commit `89339ce`）。mmdet 把 CUDA
  张量的 focal loss 派到 `mmcv.ops.sigmoid_focal_loss`，其编译核 mmcv-lite
  不带，于是每个 `FocalLoss` 头在第一次反向时死去。RTMDet 从不碰到它。
  `patch_focal_loss_for_mmcv_lite()` 把这些头转到 mmdet 自己的
  `py_sigmoid_focal_loss`，同一计算。论文自己的 FCOS/ATSS 数字来自在
  site-packages 安装里直接做的一行等价修改 — 那次修改从未进仓库。
- **一个截断的原始 IQ 场景会中止整次裁剪缓存构建**（commit `88c02ff`）。测试集
  2964 个里有 1 个写了一半的 `.npz`，在 zipfile 深处报错且不带文件名。
  `load_raw_iq` 现在回退到已解码的 `.npy` 缓存，或在无法回退时点名该场景。

## 跨检测器全扫

预测框配方不局限于 RTMDet 或 FCOS。我们把整条按检测器的链路——训练检测器、在训练/测试集上导出其预测框、
构建数量匹配（174,136）的预测框裁剪缓存、训练三个识别器（种子 101/202/303）、桥接部署——跑在**十三个检测器**
上，覆盖 anchor-free、anchor-based、自适应、dense、two-stage、多阶段、集合预测、DETR 各族。**每一个都为正**；
论文中即 `tab:taxonomy` 表，逐族数值见 `taxonomy-results.csv`。

配置为这里新增的九个 `*_stft3_memmap_resize512.py`（`cascade_rcnn`、`faster_rcnn`、`conditional_detr`、
`dab_detr`、`deformable_detr`、`dino`、`gfl`、`retinanet`、`sparse_rcnn`），加上已有的 RTMDet、FCOS、ATSS。
每个用 `_base_ = 'mmdet::...'` 继承其 mmdet 基配置并加载 `mmdet_plugins`（文档记录的 mmdet 例外）。

每个检测器 `<fam>`，从全新克隆：

```bash
# 1. 检测器（20 ep）。DETR 在统一 5e-4 下坍缩——见下面的学习率说明。
python configs/detection_is_easy/run_mmdet_train_eval.py \
  --root data/torchsig_hardshort_lowsnr_stft3_memmap/coco_multiclass \
  --config configs/detection_is_easy/<fam>_stft3_memmap_resize512.py \
  --work-dir work_dirs/<fam>_det --epochs 20 --batch-size 4 --optimizer AdamW --lr 5e-4 \
  --seed 7 --require-tensor-stats
# 2. 在测试集与训练集上导出预测框（同脚本，--eval-only --dump-results；训练集导出加 --test-split train）
# 3. 数量匹配的预测框裁剪缓存
python configs/detection_is_easy/build_pred_matched.py --fam <fam> \
  --baseline-pred work_dirs/<fam>_traindump/source_data/test_predictions.bbox.json \
  --work-dir work_dirs/<fam>_buildpred
# 4. 在该缓存上训三个识别器，再 bridge --split test（见"识别器与部署格子"）
```

**学习率偏差（记录在案，非调参优势）。** 在统一学习率 `5e-4` 下每个 DETR 变体都坍缩为零 mAP 检测器
（loss 正常但 query 退化）。按族降学习率：RetinaNet 与 deformable-attention 系 DETR（Deformable-DETR、
DINO）降到 `1e-4`；纯 DETR（Conditional-DETR、DAB-DETR）降到 `5e-5`。`run_mmdet_smoke.py` 与
`run_mmdet_train_eval.py` 里的算子回退（RoIAlign、多尺度可变形注意力、NMS 改走 `torchvision` / 纯 PyTorch）
是让 two-stage 与 DETR 检测器在无编译算子的 mmcv-lite 下能跑起来的关键。

## 出处

逐格 CSV、汇总对比，以及每次运行的 `run_info.json`（`argv` 里的字面命令行、
git commit，以及两个运行时契约开关）与本记录一起归档。论文值参考表是
`configs/detection_is_easy/paper_values.csv`；用下面的命令重生成对比：

```
python configs/detection_is_easy/collect_repro_results.py \
  --root work_dirs/repro --markdown reports/repro_cells.md \
  --reference configs/detection_is_easy/paper_values.csv
```
