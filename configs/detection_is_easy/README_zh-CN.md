# DetectionIsEasy — Detection Is Easy, Recognition Is Hard

[English](README.md) | 简体中文

宽带检测+识别研究的复现代码：

> S. Chang, Z. Yang, J. He, S. Huang, and Z. Feng, "Detection Is Easy, Recognition Is
> Hard: Rethinking Vision-Based Wideband Signal Detection and Recognition,"
> IEEE Transactions on Cognitive Communications and Networking (TCCN), under review.

配套位置：消融配置在 [`configs/detection_is_easy/`](../../configs/detection_is_easy)，
战役工具在 [`configs/detection_is_easy/`](../../configs/detection_is_easy)。

## 方法简述

宽带频谱感知被写成 STFT 谱图上的目标检测。两件事驱动全文。第一，定位已经饱和：
视觉检测器在已发布基准上的类无关框 mAP 约 0.893 — 找到信号很容易。第二，细粒度
识别才是缺口：57 类的类感知 mAP 只有约 0.45，因为谱图没有用好携带调制身份的
相位。论文沿输入表示、相位效用、检测器复杂度与检测器家族消融纯视觉配方，再加
一条领域匹配的回到 IQ 分支：标成星座家族（PSK/ASK/QAM）的框被信道化回基带 IQ，
再用一维层次识别器重分类，用 GT 框训练的识别器把部署 mAP 提高 +0.028，改在检测器
自己的预测框上训练后提高 +0.093 — 起决定作用的是识别器的训练预算与训练框分布，
不是架构。

## 论文章节 → 代码对照

| paper | code |
|---|---|
| Detector ablation grid (input rep / complexity / family) | `configs/detection_is_easy/rtmdet_*`, `fcos_*`, `atss_*`, `yolox_*`, `faster_rcnn_*`, `cascade_rcnn_*`, `deformable_detr_*` |
| STFT / raw-IQ Load transforms, complex data preprocessors, complex-1D backbone | `configs/detection_is_easy/mmdet_plugins.py` |
| Complex-1D primitives + analytic filterbanks | `configs/detection_is_easy/iqdet_complex.py` |
| Return-to-IQ recognizer backbone (1-D ResNet, iq/diff/iqdiff) | `csrr/models/backbones/returniq_resnet1d.py` |
| Hierarchical AMC head (coarse router + 45-class single + 12-class OFDM) | `csrr/models/heads/hierarchical_amc_head.py` |
| Channelized-crop dataset (57-class, `*_L1024.npz` caches) | `csrr/datasets/wideband_channelized.py` |
| Recognizer training recipe (120 ep AdamW + cosine + EMA + label smoothing) | `configs/detection_is_easy/returniq_resnet1d_{iq,diff,iqdiff}_120e_wideband.py` |
| Detect → channelize → recognize bridge, oracle bounds, diagnostics | `configs/detection_is_easy/bridge.py` |
| Class-aware detection mAP + time-frequency IoU metrics | `configs/detection_is_easy/iqdet_metrics.py` |
| Wideband data generation (TorchSig) + COCO export + memmap packing | `configs/detection_is_easy/prepare_torchsig_iq_stratified.py`, `export_*_coco_from_raw.py`, `make_stft_feature_tensor_from_complex.py`, `pack_coco_tensors_to_memmap.py`, `build_multiclass_coco.py` |
| Paper figures + corrected block-SNR analysis | `configs/detection_is_easy/make_figs.py`, `render_example.py`, `analyze_snr_stratified.py`, `analyze_box_quality.py` |

## 环境

```bash
pip install -r requirements/detection_is_easy.txt
```

该文件钉死了论文所用版本（torch 2.7.1+cu128，numpy 2.2.6，mmdet 3.3.0，
mmengine 0.10.7，torchsig 2.1.1），环境是 Ubuntu、8×RTX 4090。

**有一个选择会改你的数字：装哪一种 mmcv。** 所有报告结果都来自 `mmcv-lite` —
没有编译 `_ext` CUDA 算子的 mmcv。工具会检测到缺失扩展，并装上纯 PyTorch 的
NMS 回退（`run_mmdet_smoke.py` 里的 `maybe_stub_mmcv_ext()`），每次运行都在
`run_info.json` 里记 `used_mmcv_lite_stub: true`；带这个字段的 268 次运行全部
为真。装完整 CUDA mmcv 受支持也更快，但会换成另一套 NMS 实现，预期会有小差异。
选定一种后，整组对比都不要换。

同一个缺失扩展也拿掉了 CUDA focal-loss 核。RTMDet 不会察觉 — 它的分类损失是
纯 PyTorch — 但每个 `FocalLoss` 头（FCOS、ATSS、RetinaNet）没有它就会在第一次
反向时死掉。`patch_focal_loss_for_mmcv_lite()` 把这些头转到 mmdet 自己的
`py_sigmoid_focal_loss`，算的是同一个量，每次运行在 `run_info.json` 里用
`used_pytorch_focal_loss` 记录回退是否生效。

`torchsig` 只在再生成数据集时需要。这里的版本钉很重要：生成器的类别体系就是
那 57 类。

## 数据

来自 TorchSig 的合成宽带捕获，用自定义配置生成：
**50 000 / 5 000 / 10 000 训练/验证/测试场景**（合计 65 000 — 目录名里的 “65k”
是总数，不是训练集），每场景 262 144 个复采样、10 MHz，每场景 1–6 个信号，
57 类。

这个配置有两个性质让任务变难，两者都写在数据集目录名里。`hardshort`：信号时长
是观测的 0.5%–25%，所以每次发射只占谱图一小块。`lowsnr`：逐信号 SNR 从 −20 dB
抽到 +10 dB，分成三个等量桶。

生成资产很大（打包 STFT memmap 约 191 GB，原始 IQ 约 128 GB），不随仓库发布。
六条命令按论文精确参数重建：

```bash
cd <repo-root>
DATA=data                       # or an NVMe scratch path
RAW=$DATA/torchsig_hardshort_lowsnr_iq_65k_nvme
MM=$DATA/torchsig_hardshort_lowsnr_stft3_memmap

# 1) raw IQ scenes + per-signal metadata  (the slow step)
python configs/detection_is_easy/prepare_torchsig_iq_stratified.py \
  --out-root $RAW \
  --train 50000 --val 5000 --test 10000 \
  --num-iq-samples 262144 --sample-rate 10000000 \
  --num-signals-min 1 --num-signals-max 6 --impairment-level 0 \
  --fft-size 512 --stft-fft 512 --stft-hop 512 \
  --duration-min-frac 0.005 --duration-max-frac 0.25 \
  --bandwidth-min-frac 0.0125 --bandwidth-max-frac 0.49 \
  --center-freq-min-frac -0.45 --center-freq-max-frac 0.45 \
  --snr-buckets '-20,-10;-10,0;0,10' \
  --cochannel-overlap-probability 0.35 \
  --fast-snr-update \
  --seed 20260640

# 2) complex STFT tensors [2,F,T] + COCO annotations
python configs/detection_is_easy/export_complex_stft_coco_from_raw.py \
  --src-root $RAW --out-root $MM --stft-fft 512 --stft-hop 512

# 3) 3-channel [real, imag, log-magnitude] feature tensors  (SEPARATE --out-root)
python configs/detection_is_easy/make_stft_feature_tensor_from_complex.py \
  --src-root $MM/coco --out-root ${MM}_stft3 --mode realimag_logpower3ch --workers 8

# 4) pack into the memmap the fast training path reads
python configs/detection_is_easy/pack_coco_tensors_to_memmap.py \
  --kind tensor --src-coco ${MM}_stft3/coco --out-root $MM --splits train,val,test --workers 8

# 5) single-class ("signal") annotations -- the class-agnostic localization task
python configs/detection_is_easy/export_raw_coco_from_metadata.py \
  --src-root $RAW --out-root $MM --single-class

# 6) 57-class annotations -- the class-aware task
python configs/detection_is_easy/build_multiclass_coco.py \
  --dataset-dir $MM --out-dir $MM/coco_multiclass/annotations --splits train,val,test
```

之后，`$MM/coco/` 里是单类标注（配合 `--root $MM/coco`），`$MM/coco_multiclass/`
里是 57 类标注（其余全部用它）。

### 这条链上五件会悄悄毁掉基准的事

**SNR 范围和桶是同一个选择，不是两个。** 传 `--snr-buckets` *或*
`--snr-db-min/--snr-db-max`，不要传互相冲突的一对 — 工具现在会在不匹配时中止。
更早的修订会无条件从桶推导范围，所以只传 `--snr-db-min -20 --snr-db-max 10`
而不传桶时，会静默按*默认* −10…+40 dB、五个桶生成。那是另一套容易得多的基准，
而且没有报错、没有警告。若只给范围，桶会按 `--snr-num-buckets`（默认 3）等分；
解析后的计划会在启动时打印 — 请读它。

**`--fast-snr-update` 会同时改物理和随机流。** 它用命令的时域功率缩放替换
TorchSig 的逐信号谱图精修，并从数据集生成器的 RNG 抽 `snr_db`。论文用了它
（`summary.json` 记着 `fast_snr_update: true`）。省略它会在同一种子下得到
另一份数据集。

**第 3 步绝不能写进自己的输入。** 若 `--src-root $MM/coco --out-root $MM`，
输出会解析回 `$MM/coco`，这一步会把正在读的 `[2,F,T]` 张量覆盖成 `[3,F,T]` —
不可恢复，再跑时每个已转换文件都会失败。工具现在会拒绝；请像上面那样给单独的
`--out-root`。

**第 6 步把 `coco/<split>/` 链进多类根，这条链接是承重的。** 工具靠测试
`<root>/<split>/tensors` 是否存在来选 `tensors/` 数据前缀，memmap 加载器再从
这条路径读回划分名。一个只装 `annotations/` 的多类根会把划分解析成 `images`，
然后报 `FileNotFoundError: .../memmap/images.npy`。

**复现等级：`statistical`。** 公开流程保留协议与生成器设置，但不能重新生成
论文使用的同一份语料 realization。

**分片没有随仓库发布。** 论文语料是十个分片再用硬链接合并（`summary.json` 记着
`merge_mode: hardlink` 与 `source_shards: shard_000..009`）；产出并合并它们的
驱动不在本仓库，逐片种子也从未记录。上面的命令是整块生成。结果是同一分布、
同一生成器设置下的语料，**不是同一份实现** — 十分片运行与单次运行消耗 RNG
的方式不同。因此验收是统计上的，靠下面的等价检查，而不是校验和。

### 训练前先检查你生成了什么

```bash
python configs/detection_is_easy/validate_coco.py --root $MM
```

至少确认：57 个类别、id 0–56 在各划分上相同；50 000/5 000/10 000 张图；
逐信号 `snr_db` 覆盖 −20…+10 dB 而不是 −10…+40；框宽高与
`duration_frac ∈ [0.005, 0.25]`、`bandwidth_frac ∈ [0.0125, 0.49]` 一致；
`summary.json` 带三通道的 `stft_tensor_stats`；以及 `memmap/<split>.npy`
行数等于标注图像数。`summary.json` 还记有 `provenance` 块（torchsig 版本、
git commit、种子、argv）— 报告复现时请引用它。

若数据不在 `<repo-root>/data/` 下，请给训练工具传 `--memmap-root` / `--raw-root`，
并为 `bridge.py` 设置 `IQDET_MEMMAP_ROOT` / `IQDET_RAW_ROOT` / `IQDET_CACHE_ROOT`。
只改 `--root` 只会移动标注。

## 复现一个数字

三个阶段。每一阶段产出下一阶段需要的输入。

### 阶段 1 — 检测器

**训练和导出预测是两次调用。** `--dump-results` 只设置测试评估器的输出前缀；
训练调用从不跑测试循环，所以不会写预测。先训练，再对 checkpoint 用
`--eval-only` 调一次：

```bash
# train (this is the deployment detector: the run every bridged number is computed from)
python configs/detection_is_easy/run_mmdet_train_eval.py \
  --root $MM/coco_multiclass \
  --config configs/detection_is_easy/rtmdet_m_stft3_tensor_memmap_resize512.py \
  --work-dir work_dirs/baseline_mc_rtmdet_m_20e_seed20262811 \
  --epochs 20 --batch-size 8 --optimizer config --seed 20262811

# then dump test predictions from the trained checkpoint
python configs/detection_is_easy/run_mmdet_train_eval.py \
  --root $MM/coco_multiclass \
  --config configs/detection_is_easy/rtmdet_m_stft3_tensor_memmap_resize512.py \
  --work-dir work_dirs/baseline_mc_rtmdet_m_20e_seed20262811_testdump \
  --eval-only --checkpoint work_dirs/baseline_mc_rtmdet_m_20e_seed20262811/epoch_20.pth \
  --dump-results
# -> work_dirs/..._testdump/source_data/test_predictions.bbox.json
```

`--work-dir` 必填，没有默认值。`--optimizer config` 保留配置自己的 AdamW
（lr 1e-4）；若改成 `--optimizer AdamW --lr 5e-4` 会选*另一套*配方 — 见下表，
两列分别是 “uniform recipe” 与 “own schedule”。

### 阶段 2 — 识别器

先缓存信道化裁剪，再训练。论文的识别器由 `bridge.py` 训练，这样才能产出
`bridge.py bridge` 能加载的 checkpoint：

```bash
for s in train val test; do
  python configs/detection_is_easy/bridge.py build --split $s --L 1024
done   # -> work_dirs/returniq_cache/{train,val,test}_L1024.npz

python configs/detection_is_easy/bridge.py train-hier \
  --train-cache work_dirs/returniq_cache/train_L1024.npz \
  --val-cache   work_dirs/returniq_cache/val_L1024.npz \
  --out work_dirs/returniq_cache/recognizer_hierrcpA_s101.pth \
  --epochs 120 --label-smooth 0.1 --cosine --ema 0.999 --aug-cfo 0.02 --seed 101
```

**`train-hier` 的默认值是论文的负结果，不是标题结果。** 它们是 40 个 epoch、
无 label smoothing、无 cosine、无 EMA — 正是那个训练不足的识别器，差点被当成
结构上限发表（combined 干净准确率 0.534，对上面配方的 0.875；这一对最初报告为
0.643 对 0.869，把一个 stage2-single 数字和一个 combined 数字混在了一起）。那条
命令里的五个开关才是发现。

同一识别器也作为一等 CSRR 模型提供，按通常方式训练：

```bash
python tools/train.py configs/detection_is_easy/returniq_resnet1d_iq_120e_wideband.py
```

用这条路径在 CSRR 里研究架构。用 `bridge.py train-hier` 复现论文：两者保存的
checkpoint 格式不同，`bridge.py bridge` 读的是后者。

### 阶段 3 — 部署桥

```bash
python configs/detection_is_easy/bridge.py bridge \
  --split test \
  --baseline-pred work_dirs/baseline_mc_rtmdet_m_20e_seed20262811_testdump/source_data/test_predictions.bbox.json \
  --hier-model work_dirs/returniq_cache/recognizer_hierrcpA_s101.pth \
  --L 1024 --score-thr 0.05 --limit 2963 --class-nms-iou 0.5 \
  --ours-score-recog --iq-families psk,ask,qam
```

**不要用默认值跑这一条。** 其中四个开关与默认不同，单独改任何一个都会改答案：

| flag | default | paper | what the default does |
|---|---|---|---|
| `--score-thr` | `0.0` | `0.05` | keeps near-zero-score detections, flooding both methods |
| `--limit` | `0` (all) | `2963` | scores a different scene set, so numbers are not comparable to the paper |
| `--class-nms-iou` | `1.0` | `0.5` | **disables** per-class NMS; duplicate boxes inflate both sides asymmetrically |
| `--ours-score-recog` | off | on | ranks routed detections by detection score alone, discarding recognition confidence |

`oracle` 给出完美框上界；`diag-quality` 写出构建 Fig. 2 所用的逐检测转储。

## 哪一格对应什么：配置、开关、种子、期望值

每个标题数字，以及产出它的东西。除非另注，mAP 是 **val** 划分上的
`coco/bbox_mAP`。所有检测器行使用 `--root $MM/coco_multiclass` 与
`run_mmdet_train_eval.py`；“uniform” = `--optimizer AdamW --lr 5e-4`，
“own” = `--optimizer config`。

类无关（只做定位）任务不需要单独配置：把 `--root` 指到单类标注（`$MM/coco`），
工具会从中读出一个类别并相应设置 `num_classes`。

**怎么读 “expected”。** 每个检测器期望值都是 [`REPRODUCTION_zh-CN.md`](REPRODUCTION_zh-CN.md)
记录的端到端重跑给出的 **3 种子均值 ± sd**，也是稿件现在报告的值；单次运行会落在
它的 ±0.023 以内（测得的同种子地板的三倍）。论文只用一个种子陈述的格子列为单值。
最后一列保留论文*最初*报告的值，让历史不丢失；哪里不同，`REPRODUCTION_zh-CN.md`
说明原因。

| paper cell | config | recipe / flags | seeds | expected (3-seed mean ± sd) | tol. | originally reported |
|---|---|---|---|---|---|---|
| Localization is easy (class-agnostic) | `rtmdet_m_stft3_tensor_memmap_resize512.py` | uniform, `--batch-size 8`, **`--root $MM/coco`** | any, 3 seeds | 0.893 ± 0.006 | ±0.023 | 0.948（更容易的生成器配置） |
| Tab. I tiny / uniform | `rtmdet_tiny_stft3_tensor_memmap_resize512.py` | uniform, `--batch-size 8` | any, 3 seeds | 0.432 ± 0.004 | ±0.023 | 0.431 |
| Tab. I small / uniform | `rtmdet_s_stft3_tensor_memmap_resize512.py` | uniform, `--batch-size 8` | any, 3 seeds | 0.443 ± 0.021 | ±0.023 | 0.449 |
| Tab. I medium / uniform (= Tab. III STFT3 offline reference) | `rtmdet_m_stft3_tensor_memmap_resize512.py` | uniform, `--batch-size 8` | any, 3 seeds | 0.472 ± 0.012 | ±0.023 | 0.460（Tab. III 的参考格为修订时新增） |
| Tab. I large / uniform | `rtmdet_l_stft3_tensor_memmap_resize512.py` | uniform, `--batch-size 8` | any, 3 seeds | 0.451 ± 0.007 | ±0.023 | 0.462 |
| Tab. I tiny / own | `rtmdet_tiny_stft3_tensor_memmap_resize512.py` | own, `--batch-size 4` | any, 3 seeds | 0.433 ± 0.004 | ±0.023 | 0.408（2 种子） |
| Tab. I small / own | `rtmdet_s_stft3_tensor_memmap_resize512.py` | own, `--batch-size 4` | any, 3 seeds | 0.470 ± 0.014 | ±0.023 | 0.429 |
| Tab. I medium / own **(deployment detector)** | `rtmdet_m_stft3_tensor_memmap_resize512.py` | own, `--batch-size 4` | any, 3 seeds | 0.492 ± 0.010 | ±0.023 | 0.477 ± 0.039（混入 batch-8 运行 0.521） |
| Tab. I large / own | `rtmdet_l_stft3_tensor_memmap_resize512.py` | own, `--batch-size 4` | any, 3 seeds | 0.504 ± 0.015 | ±0.023 | 0.486（2 种子） |
| Tab. III magnitude-only (phase out) | `rtmdet_m_rawiq_fourier_logmag2ch_resize512.py` | uniform, `--batch-size 6` | any, 3 seeds | 0.441 ± 0.023 | ±0.023 | 0.455 |
| Tab. III phase + magnitude | `rtmdet_m_raw_iq_filterbank_hardshort_resize512.py` | uniform, `--batch-size 6` | any, 3 seeds | 0.455 ± 0.013 | ±0.023 | 0.447（2 种子） |
| Tab. III phase only | `rtmdet_m_rawiq_fourier_realimag_resize512.py` | uniform, `--batch-size 6` | 7 | 0.431 | ±0.023 | 0.440 |
| Tab. III learnable filterbank | `rtmdet_m_rawiq_learnable_realimag_logmag_resize512.py` | uniform, `--batch-size 6` | 7 | 0.418 | ±0.023 | 0.412 |
| Tab. III complex-1D + FFT bridge | `rtmdet_m_complexiq1d_fftbridge_resize512.py` | uniform, `--batch-size 6` | 7 | 0.053 | collapse | 0.026 |
| §VI-B FCOS | `fcos_stft3_memmap_resize512.py` | uniform, `--batch-size 4`, **`--require-tensor-stats`** | any, 3 seeds | 0.470 ± 0.005 | ±0.023 | 0.374（恒等归一化，首轮） |
| §VI-B ATSS | `atss_stft3_memmap_resize512.py` | uniform, `--batch-size 4`, **`--require-tensor-stats`** | any, 3 seeds | 0.468 ± 0.003 | ±0.023 | 0.380（恒等归一化，首轮） |
| §VI-B RTMDet-M, matched batch | `rtmdet_m_stft3_tensor_memmap_resize512.py` | uniform, `--batch-size 4` | any, 3 seeds | 0.426 ± 0.001 | ±0.023 | —（修订时新增） |
| §IV-A recognizer, recipe A | — | `train-hier --epochs 120 --label-smooth 0.1 --cosine --ema 0.999 --aug-cfo 0.02` | 101/202/303 | 0.875 combined clean accuracy | ±0.006 | 0.869 |
| §VI-D recognizer, 40-epoch predecessor | — | `train-hier` **defaults** (`--epochs 40 --aug-cfo 0.02`) | 101 | 0.632 stage2-single / 0.534 combined | ±0.01 | 0.643（stage2-single，被标成 combined） |
| §VI-D deployment, vision → routed | — | the Stage-3 command above | 101/202/303 | +0.028 fused delta（复现视觉基线 0.474） | ±0.002 on the delta | 0.522 → 0.546 (+0.024) |
| §VI-D per-family PSK / ASK / QAM | — | same command | 101/202/303 | +0.153 / +0.132 / +0.081 | ±0.011 / ±0.008 / ±0.012 | +0.143 / +0.118 / +0.084 |
| §VI-D oracle (perfect box) | — | `oracle --with-oracle --limit 2000 --score-thr 0.05` | 101 | 0.420 → 0.608 | ±0.01 | 0.608（未变） |
| §VII recognizer trained on the detector's own predicted boxes | — | 按 `--score-thr 0.1` 建预测框裁剪缓存（`build_pred_matched.py`），用 recipe-A 的 `train-hier` 开关训练，再跑阶段 3 命令 | 101/202/303 | RTMDet 上 +0.093 ± 0.001 fused delta；FCOS 上 +0.189 ± 0.002 | ±0.002 on the delta | −0.019（缓存由高分框构建） |

关于这张表有三点要注意。

部署检测器与它的 batch 大小：上面阶段 1 的命令按 `--batch-size 8` 训练，这是本
README 里每个桥接数字所依据的配置。它的单次运行最初报告为 0.521，并在这里列为
medium/own 格；那次单次运行的复现值是 0.472，稿件已不再引用它。medium/own 格现在
与 own-schedule 列其余各格一样，按 `--batch-size 4`、三种子陈述（0.492 ± 0.010）；
最初报告的 3 种子均值 0.477 ± 0.039 把那次 batch-8 运行和两次 batch-4 运行混在了
一起。复杂度结论仍落在 uniform 列上。

定位行：论文最初把类无关定位写成 0.948。那个值测在更早、更容易的生成器配置上
（信号时长为观测的 5–100%，SNR −10 到 +20 dB，无同信道重叠），而不是这里发布的
hardshort-lowsnr 基准（时长 0.5–25%，SNR −20 到 +10 dB，35% 同信道重叠）。
在本基准上三次单类运行给出 0.893 ± 0.006，这就是稿件现在报告的值（本 README 的
早先版本曾把单次期望写成 ≈0.915）。结论不变 — 定位无论哪种都已饱和，Fig. 2 的
定位召回 ≈0.99 也是在本基准测试划分上测的 — 但跑上面的命令时期望 ≈0.89，而不是
0.948。

FCOS/ATSS 行：首轮的 0.374 / 0.380 是用恒等归一化训练的（`mean=[0,0,0] std=[1,1,1]`，
而数据的逐通道 sigma 大约 12.8），它们对照的 RTMDet 运行却用了实测统计。注入统计
（`--require-tensor-stats`）后它们达到 0.470 ± 0.005 / 0.468 ± 0.003，高于同 batch 的
RTMDet-M（0.426 ± 0.001）；稿件报告的是这些值，0.374 / 0.380 只作为历史保留。

## 两种不同指标都叫 “class-aware mAP”

它们不可互换，混用是最快得出“论文错了”的办法。

- **`coco/bbox_mAP`** — mmdet 的 `CocoMetric`，对 57 类平均，在 **val** 划分上。
  这是检测器消融指标：0.472、0.492、0.441，以及上表其余每一格。
- **`class_aware_detection_map`** — `iqdet_metrics.py` 里的时频 IoU 指标，
  在前 2963 个 **test** 场景上计算。这是部署指标：在复现视觉基线 0.474 上的
  +0.028 融合增量（最初报告为 0.522 → 0.546）。

部署基线（复现 0.474；最初 0.522）与它所桥接的检测器的 val mAP（那次运行复现
0.472；最初 0.521）接近只是巧合。它们是不同划分上的不同指标。

## 什么算成功复现

训练**不是**确定性的：工具设置 `randomness = dict(seed=..., deterministic=False)`，
所以 cuDNN 会选非确定性核。即使同一种子、同一机器，检测器重跑也落在
±0.023 带内的类感知 mAP；跨种子散布更大（上面各格带着测得的三种子标准差）。

复现的是稳健的*结论*，而不是第三位小数：

1. 类无关定位 ≈ 0.89，而 57 类 mAP ≈ 0.45。这个缺口就是论文。
2. 仅幅度与相位+幅度打平。谁也不比种子散布赢更多。
3. 可学习前端输给冻结前端。
4. 固定配方下，mAP 从 tiny → large 是平的。
5. 把 PSK/ASK/QAM 框路由回 IQ，用 GT 框训练的识别器总体约 +0.03、PSK 上约 +0.15；
   改在检测器自己的预测框上训练后总体约 +0.09。
6. 识别器的 120-epoch 配方比 40-epoch 前任高出约 +0.34 combined 干净准确率
   （0.534 → 0.875；按 stage2-single 指标约 +0.24，0.632 → 0.867）。这是预算效应，
   不是架构效应。
7. 匹配 batch、用实测归一化时，检测器家族分不开（FCOS 0.470、ATSS 0.468、
   RTMDet-M 0.426）。

若 (1)–(7) 成立，即使个别格子在第二位小数上不同，复现也算成功。

## 图

```bash
python configs/detection_is_easy/make_figs.py            # Figs. 1, 2, 4, 5 -> figs/*.pdf
python configs/detection_is_easy/render_example.py \
  --ann $MM/coco_multiclass/annotations/instances_test.json \
  --raw $RAW/raw/test \
  --pred work_dirs/baseline_mc_rtmdet_m_20e_seed20262811_testdump/source_data/test_predictions.bbox.json
```

`make_figs.py` 是自包含的：它只读旁边已入库的 `snr_data.csv`，那是
`analyze_snr_stratified.py` 在 recipe-A 诊断转储上的输出。用下面的命令重生成
该 CSV：

```bash
python configs/detection_is_easy/bridge.py diag-quality \
  --hier-model work_dirs/returniq_cache/recognizer_hierrcpA_s101.pth \
  --baseline-pred <the test dump> --L 1024 --score-thr 0.05 --with-oracle --limit 2000 \
  --out work_dirs/returniq_cache/box_quality_oracle_rcpA.jsonl
python configs/detection_is_easy/analyze_snr_stratified.py \
  --jsonl work_dirs/returniq_cache/box_quality_oracle_rcpA.jsonl --limit 2000
```

## 已记录的偏差 / 说明

- **块 SNR 校正。** 所有按 SNR 分层的结果使用
  `block_snr = snr_db + 10*log10(1/(tf*ff))`，其中 `tf` 与 `ff` 是信号的时间与
  频率占用。生成器的 `snr_db` 是整段观测平均，会把可见度低估中位数约 +16.7 dB。
  不要在原始轴上把结果标成 “low-SNR”。
- **mmcv `_ext` 桩。** 见环境一节：论文数字来自纯 PyTorch NMS 回退，每次运行
  记在 `run_info.json`。
- **归一化统计。** 原始 IQ 滤波器组配置带着离线 STFT3 统计的逐通道均值/标准差，
  而不是在自己的前端输出上重算。它们一开始是占位，从未改过，所以那些格子报告的
  每个数字都是用这套常数训出来的。把它们当作配方的一部分，不要重算：相位试验
  的两臂共用同一组常数，比较才公平，但改它们就会改数值。
- **定位数字。** 见复现表下的提醒：最初报告的 0.948 来自更早、更容易的生成器
  配置；本基准上三次单类运行给出 0.893 ± 0.006，这就是稿件现在报告的值。
- **FCOS/ATSS 归一化。** 首轮的 FCOS/ATSS 值（0.374 / 0.380）是用恒等归一化训练的；
  这两个头务必传 `--require-tensor-stats`。传了之后它们达到 0.470 ± 0.005 /
  0.468 ± 0.003，即稿件报告的值。
- **合成来源。** 类别、框和 SNR 都是生成器真值；没有测量噪声底可以躲 — 识别缺口
  是结构性的，发布的配置让每个数字都可以再生成。

Licensed under the Apache License, Version 2.0.
