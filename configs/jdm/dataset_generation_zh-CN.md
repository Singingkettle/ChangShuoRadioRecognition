# CSRD（twc 配置）数据集再生成 — 噪声修复与协议

[English](dataset_generation.md) | 简体中文

配套文档：同目录 [`README_zh-CN.md`](README_zh-CN.md)（JDM 方法与模型侧预期）。
本文件记录**数据侧**：为何要再生成数据集、“噪声被重复添加”（SNR 重复添加噪声）
缺陷的根因、修复、生成协议，以及经验 SNR 核验。

## 1. 生成器

- 仓库：<https://github.com/Singingkettle/ChangShuoRadioData>，`twc/` 目录
  （TWC 论文的仿真代码，DOI `10.1109/TWC.2024.3450972`）。
- 噪声策略修复：commit `78b086b`
  `fix(twc): store noise exactly once per frame to prevent repeated noise stacking`
  （上游 `twc/` 在 `3d38a7d` 引入）。
- 工具链：带 Communications / Signal Processing / DSP System 工具箱的 MATLAB；
  无界面 `matlab -batch`。

## 2. 重复加噪缺陷的根因

生成器仓库的 git 历史里审计过三代脚本：

1. **原始脚本** `ref/DataSimulationTool/generate.m` @ `edb0323`
   （2024-01）。在 AWGN 段（第 116 行）**每一个**子信号都各自调用
   `awgn(new_sub.data, dB)`。接收帧是子信号的*和*，于是 N 个信号的帧会累加 N 次
   独立噪声：噪声功率 ×N，即有效 SNR = 标签 − 10·log10(N)（4 个信号时为 −6 dB）。
   这就是字面意义上的“SNR 段里噪声被重复添加”。
2. **加了守卫的修订** @ `0241d26`（2024-05）— 产出 2024-05 磁盘导出的版本。
   它在 `awgn` 调用外包了 `if sub_signal_index == 1`，所以宽带求和只带一次噪声
   （我们经验核验过，见 §5）。但又引入/保留了另外三个缺陷：
   - `real`/`real_awgn` 版本：子信号 2..N 被保存时**没有衰落信道**
     （`c(...)`/时钟偏移输出算过，又因为 `new_sub.data` 只在
     `sub_signal_index == 1` 分支里赋值而被丢掉）；
   - v104（`real`）：`awgn` 加在**过信道之前**的信号上，于是子信号 1 整段
     Rician/Rayleigh + 时钟偏移处理被丢弃；
   - 不带 `'measured'` 的 `awgn(x, dB)` 假定输入功率为 1 — 过衰落信道后参考错了，
     所以 `real_awgn` 的 SNR 标签与数据对不上（核验：过信道后子信号 1 的总功率
     在 0.33–9.8 之间波动，而预期应是 1 + 噪声）。
3. **当前上游 `twc/generate.m`** @ `3d38a7d`（2026-02）引入了
   `add_wideband_awgn`（噪声在宽带层实现一次，`wideband_data` 正确），
   **但仍把同一份噪声向量加进每个子信号保存的 `signal_data`**（修复前
   第 120–123、180–183、219–223 行）。工具箱既定的消费路径是把 `signal_data`
   求和组成帧（历史 `tools/convert_datasets/cache_csrr.py` 第 36 行
   `np.sum(x, axis=0)`；当前是 `LoadCSRDFrame`），于是同一噪声向量叠 N 次 →
   噪声幅度 ×N → **噪声功率 ×N²**，有效 SNR = 标签 − 20·log10(N)
   （4 个信号时为 −12 dB）。还发现第二个潜伏 bug：`add_clock_offset` 用
   `interp1`，当时钟因子 C < 1 时会用 NaN 外推尾部；一个 NaN 污染功率计算，
   导致 `real`/`real_awgn` 版本保存的 `wideband_data` **全为零**
   （在未修复代码的冒烟运行中确认）。

## 3. 修复（生成器 commit `78b086b`）

`twc/` 里的最小改动：

- `generate.m`：帧的那一次 AWGN 实现**只**存在 `wideband_data` 里
  （= 过信道子信号之和 + 噪声）。`signal_data` 现在保存无噪声的过信道子信号，
  任何消费者都无法通过求和把噪声叠上去。SNR 标签含义不变：各子信号功率
  （对该帧子信号取平均）除以宽带噪声总功率。
- `add_clock_offset.m`：`interp1` 产生的 NaN 尾部改成零。
- `generate.m` 变成函数 `generate(num_items, output_root)`，并用
  `rng(0, 'twister')` 保证可复现（默认保持旧行为）。

本仓库消费侧的改动（与本文同一提交）：
`csrr/datasets/transforms/csrd.py::LoadCSRDFrame` 现在优先读
`wideband_data`，只有没有它时才回退到对 `signal_data` 求和
（对无噪声配置以及旧导出是正确的，旧导出里那一次噪声在子信号 1 里）。

## 4. 生成协议（按论文 / twc 配置）

| parameter | value |
|---|---|
| sample rate | 150 kHz |
| frame length | 1200 samples (12000 synthesized, decimated ×10 by design) |
| modulations | BPSK, QPSK, 8PSK, 16QAM, 64QAM |
| samples per symbol | {10, 12, 15} (bandwidth diversity) |
| signals per frame | recursive placement with protect gap 2·BW, ≈2–5 |
| SNR grid | −8:2:30 dB (20 levels) |
| channel configs (124 versions) | v1 ideal; v2–v71 Rician (7 speeds × K=1..10); v72–v78 Rayleigh (7 speeds); v79–v98 AWGN (20 SNRs); v99–v103 clock offset (max 1,3,5,7,9 ppm); v104 "real" (random fading+offset+SNR); v105–v124 "real_awgn" (fading+offset, fixed SNR each) |
| frames per version | 1000 |

输出：把 124×1000 的导出写到 `data/ChangShuoTwc2026`（再软链接进 CSRR 仓库）。
2024 年的导出请另放，用来对比 SNR；不要把两套混在同一个 `data_root` 下。

每个版本的布局（`csrr/datasets/csrd.py` 消费的 schema）：

```
v<k>/
  anno/000001.json ... 001000.json     # per-frame parallel arrays:
                                       # center_frequency, bandwidth, snr,
                                       # modulation, channel, sample_rate,
                                       # sample_num, sample_per_symbol, file_name
  sequence_data/iq/000001.mat ...      # signal_data  (num_signals, 2, 1200)  noise-FREE
                                       # wideband_data (1, 2, 1200)           received frame,
                                       #   present only for awgn-*/real/real_awgn-* versions
```

不写划分文件；`CSRDDetectionDataset` / `CSRDModulationDataset` 在加载时对每个
版本做确定的、带种子的 50/10/40 训练/验证/测试划分（见 `README_zh-CN.md`），
因此新导出**不需要转换步骤**。

启动命令（可长期使用，只吃 CPU）：

```bash
cd /path/to/ChangShuoRadioData/twc
matlab -batch "generate(1000, '/path/to/ChangShuoTwc2026')"
# then: ln -s /path/to/ChangShuoTwc2026 <csrr>/data/ChangShuoTwc2026
```

## 5. 经验 SNR 核验

方法：对有干净参考的帧，精确重建噪声并与标签比较
（`tools/misc/verify_csrd_snr.py`）。

**旧数据集**（v79 全部 1000 帧，awgn −8 dB）：每帧恰好一个子信号带噪声
（直方图 {1: 1000}）；宽带求和 SNR − 标签：均值 +0.004 dB，标准差 0.13 →
旧 AWGN 版本在磁盘上*没有*双重加噪，但 §2.2 里 `real`/`real_awgn`/v104 的缺陷
仍在；任何从 `signal_data` 再切逐信号片段的消费者都会看到不一致的噪声
（噪声全在子信号 1，其余没有）。

**新数据集**（冒烟运行，已修复生成器；测得 − 标签，单位 dB）：

| version | label | item 1 | item 2 | `signal_data` residual vs clean |
|---|---|---|---|---|
| v79 | −8 | −0.13 | +0.05 | 0 (noise-free ✓) |
| v84 | +2 | +0.02 | +0.03 | 0 |
| v89 | +12 | +0.22 | −0.05 | 0 |
| v94 | +22 | −0.16 | −0.19 | 0 |
| v98 | +30 | −0.02 | −0.07 | 0 |
| v105 (real_awgn) | −8 | +0.20 | −0.04 | n/a (measured post-channel ref) |
| v115 (real_awgn) | +12 | +0.04 | −0.02 | |
| v124 (real_awgn) | +30 | −0.05 | +0.05 | |

`add_clock_offset` 的 NaN 修复之前，`real_awgn` 行是退化的
（`wideband_data` ≡ 0）；修复后与标签相差在 ±0.2 dB 以内。
剩余 ±0.2 dB 散布是 1200 点噪声估计的自然逐次方差，不是偏差。

生成完成后，再跑一遍全量检查：

```bash
python tools/misc/verify_csrd_snr.py --data-root data/ChangShuoTwc2026
```

## 6. 状态

生成只吃 CPU。导出落到 `data/ChangShuoTwc2026` 之后，JDM 配置相对 CSRR 仓库
解析 `data_root`。
