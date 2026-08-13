# JDM (Joint Signal Detection and Automatic Modulation Classification) — Recon Notes

Working notes for the clean re-implementation of the paper method on this repo's
mmengine stack. Companion doc: `README.md` in this directory (method → code map,
train/test instructions).

## 1. The paper

**Joint Signal Detection and Automatic Modulation Classification via Deep Learning**
Huijun Xing, Xuhui Zhang, Shuo Chang, Jinke Ren, Zixun Zhang, Jie Xu, Shuguang Cui.
*IEEE Transactions on Wireless Communications*, vol. 23, no. 11, pp. 17129–17142, 2024.
DOI: `10.1109/TWC.2024.3450972`. Full text recovered from the arXiv HTML version:
<https://arxiv.org/html/2405.00736v1> (v1, May 2024). Official code/dataset pointer in the
paper: <https://github.com/Singingkettle/ChangShuoRadioData>.

### 1.1 Task and framework (JDM)

Multiple modulated signals coexist at different carrier frequencies inside one received
frame. The receiver gets a single-channel I/Q sequence `x ∈ R^{2×L}` (L = 1200). Two
sequential sub-tasks:

1. **Detection**: predict the set `{(c_i, w_i)}` of center frequency + bandwidth pairs.
2. **AMC**: for each detected band, classify the modulation (5 classes: BPSK, QPSK,
   8PSK, 16QAM, 64QAM).

JDM = two interconnected modules; the detection module hands "**proposals**"
`[f_c, B, conf]` to the classification module, which filters the raw signal down to a
single baseband signal per proposal and classifies it.

### 1.2 Detection module (YOLO-style, 1-D)

- Input: FFT of the I/Q frame (`fftshift`ed), keeping the `2×L` layout
  (amplitude + phase channels in the historical implementation).
- Backbone: 5 sequential CNN blocks, each = 3 conv layers + ReLU + BatchNorm
  (historical `DetCNN`: channel progression 16→32→64→128→256, MaxPool(2) after
  blocks 1–4, "valid" padding; the paper quotes a final feature map of
  `N×1×144×256`).
- Head: YOLO-inspired, **single scale level**. The frequency axis is divided into
  grid cells ("detection units"); each cell holds `C = 3` anchors of different base
  bandwidths. Each anchor predicts 3 attributes: center-frequency offset `f_c`
  (sigmoid, within-cell), bandwidth `B` (log-scale w.r.t. the anchor width), and a
  confidence score. **No class prediction in the detector** — it is signal/no-signal
  (class-agnostic); modulation comes from the second stage.
- Box convention: a "box" is a frequency interval `(f_c − B/2, f_c + B/2)`. The
  other (time) axis is always fully occupied, so **IoU degenerates to 1-D interval
  IoU** along the frequency axis. (Historical code confirms: boxes were stored as
  `[x, 0, w, 1]` with y fixed to the full extent and the y terms commented out of
  the coder.)
- Assignment: the grid cell containing a GT center is "responsible"; the anchor
  with the highest IoU is the positive predictor (paper: "objects are assigned to
  predictors based on the highest IoU scores").
- The 3 anchor sizes mirror the 3 bandwidth clusters of the dataset:
  `(0,110)`, `(110,130)`, `(130,150)` FFT bins (also used for AP-small/medium/large).

### 1.3 Classification module

- Input: `2×1200` I/Q of a **single** signal, obtained by (i) removing the carrier
  predicted in the proposal (down-conversion) and (ii) FIR low-pass filtering at the
  proposal bandwidth.
- Network (paper Fig. 5): 3 conv layers with ReLU + Dropout(0.5):
  `W1: 1→256 @ (1×3)`, `W2: 256→256 @ (1×3)`, `W3: 256→80 @ (2×3)` (collapses the
  I/Q axis), output `N×80×1194`; then squeeze → transpose → **Sum over the time
  axis** → `N×80` feature, followed by the modulation classification output.
  (The paper's stated stride `(2,2)` is inconsistent with its own output size
  `1194 = 1200 − 3×2`; the output size implies stride 1 / valid padding, which is
  what we implement. The 80-dim sum output requires a final linear projection to
  the 5 classes; the paper leaves this implicit.)

### 1.4 Training protocol (paper Sec. VI)

| | detection | classification |
|---|---|---|
| optimizer | Adam, lr 1e-3 | AdamW, lr 1e-3, weight-decay 5e-5 |
| batch size | 12 | 32 |
| epochs | 30 | 60 |

Trained separately (30 + 60 epochs), one Nvidia RTX 3090 Ti.

### 1.5 Evaluation metrics (paper Sec. VI-A)

- COCO-style **mAP** with 1-D IoU: mAP@[.5:.95], AP@.50, AP@.75.
- Size-binned AP: `AP_small/medium/large` with bandwidth thresholds
  `(0,110) / (110,130) / (130,150)` **samples (FFT bins)** instead of pixel areas.
- **AR** (average recall over IoU .5:.95), reported as AR@4 / AR@5 / AR@6
  (max-detections caps chosen from the dataset's signal-count distribution),
  plus size-binned AR.
- Classification: per-modulation accuracy vs SNR / K-factor / Doppler / clock
  offset; joint (end-to-end JDM) accuracy = detection + classification chained.

### 1.6 Dataset in the paper (CRML23)

Generated with the ChangShuoRadioData (CSRD) toolchain, `twc` profile. Recursive
band-filling algorithm: random bandwidths/carriers fill the observed band, 0–6+
signals per entry (4–5 most common). Parameters (Table I): sample rate 150 kHz;
SNR 12→30 dB step 2; Rician (K 1..10) / Rayleigh channels; path delays
[0, 1.8e-7, 3.4e-7], gains [0, −2, −10] dB; max Doppler 4 Hz; max clock offset 5 ppm;
5 modulations.

## 2. Historical implementation (git archaeology)

The current branch (`feature/amr-benchmark-migration`) contains no detector code —
it was removed by commit `94fad50` ("Remove unused models and mm-family
dependencies"). Two historical strata exist:

### 2.1 April 2023 stratum (mmcv 1.x-era API, `configs/rrdnn/`)

Commits `ad4c542`, `b7a8c7e`, `7ed4034`, `35e3de6`, `b395b27`, `a2efad9` (all
"fix bug in signal detector"/"update detector", Apr 2023). Files (recovered with
`git show a2efad9:<path>`):

| file | content | verdict |
|---|---|---|
| `configs/rrdnn/rrdnn_csrr2023.py` | model cfg: `BaseDetector` + `DetCNN` backbone + `SignalDetectionHead`; AdamW lr 3e-3, 100 epochs, fixed LR | reference only (old runner API) |
| `configs/rrdnn/data_csrr2023.py` | pipeline `LoadFFTofCSRR → ChannelMode → LoadCSRRTrainAnnotations → Collect`; data root `.../ChangShuo/CSRR2023` | reference for input representation |
| `csrr/models/backbone/cnnnet.py::DetCNN` | 5 conv blocks (16/32/64/128/256, 3 convs each + ReLU + BN, MaxPool(1,2) after blocks 1–4, valid padding) | **salvaged as spec** for the new `JDMDetectionBackbone` |
| `csrr/models/heads/detector_head.py::SignalDetectionHead` | YOLOv3-style head built on **mmdet** (`AnchorGenerator`, `YOLOBBoxCoder`, `GridAssigner`, `PseudoSampler`, `batched_nms`); 2 anchors (widths 120/90), stride 8; attributes = (cf, bw, conf); losses: BCE(conf), BCE(cf), MSE(bw, weight 2.0), reduction `sum` | logic salvaged, code rewritten (mmdet is no longer a dependency; several latent bugs, see below) |
| `csrr/models/methods/base_detector.py` | old-style `forward_train/forward_test` wrapper + a mmdet `DetDataPreprocessor` | rewritten (`SignalDetector` on mmengine `BaseModel`) |
| `csrr/datasets/csrr.py::CSRRDataset` | built a COCO-style json on the fly (`bbox=[x, 0, w, 1]`), evaluated with `pycocotools` | replaced by a native mmengine dataset + pure-numpy 1-D metric |
| `csrr/datasets/pipeline/loading.py::LoadFFTofCSRR` | `loadmat(...)['signal_data']` → **sum the per-signal components** → FFT → fftshift → stack(|X|, angle) | salvaged as spec for `LoadCSRDFrame`/`IQToSpectrum` |
| `csrr/datasets/pipeline/loading.py::LoadCSRRTrainAnnotations` | cf/bw → bins: `x = (cf/sr + 0.5)·N`, `w = bw/sr·N`, boxes `cxcywh` with y=(0.5,1) | salvaged as spec for `LoadCSRDDetectionAnnotations` |

Known defects in the old head (why it is *not* copied verbatim):

- Grid geometry inconsistent: valid-padding backbone yields a 67-cell grid for
  L=1200, while the anchor generator assumed stride 8 (centers 8, 16, …, 536) — the
  grid covered less than half of the spectrum, and `responsible_flags` could index
  out of bounds for GT centers beyond bin 536.
- Anchor y-extent hacks (`centers=(8.0, 0.5)`, boxes `(…, 0, …, 1)`) to force 2-D
  mmdet machinery to behave 1-D; the bbox coder decodes `x1==x2` in one branch
  (`x_centers[...,0] - ws[...,0]` twice) — i.e. the "1-D via degenerate 2-D"
  approach was fragile.
- `GridAssigner(pos_iou_thr=0.95, neg_iou_thr=0.95)` marks *nearly every* anchor
  negative, and positives rely only on the responsible-cell override.
- Head assumed `num_classes=0` and reused a scores hack
  (`scores=[1-obj, obj]`, `labels[...]=0`) to fit mmdet's multiclass NMS.

### 2.2 March 2026 stratum (initial commit of the current repo)

Commit `0222d01` still carried `csrr/models/heads/detector_head.py` (same
mmdet-based head, mmdet 3.x imports) and an **empty** `configs/rsnn/rsnn_csrr2023.py`;
both were deleted in `94fad50`. No dataset class or backbone for detection
survived into this stratum. Conclusion: the git history provides the *spec*
(input representation, box convention, target/loss recipe, thresholds), but no
directly reusable mmengine-2.x code. Everything below is a clean-room rewrite.

## 3. Dataset on disk

`the 2024-05 CSRD export/` (symlinked into the repo as
`data/ChangShuo`), 124 version directories `v1 … v124`, ≈69 MB each (~8.5 GB total).
This is the CSRD/`twc`-style output matching the paper's CRML23 recipe:

- Each version = one channel/impairment configuration × 1000 entries:
  `v1` ideal; `v2–…` `rician_speed_*` (70 dirs); `rayleigh_*` (7); `awgn-*dB` (20);
  `clockOffset_*` (5); `real_awgn-*dB` (21). SNR tags from `infdB` down to
  specific dB values.
- Layout per version: `anno/000001.json … 001000.json` +
  `sequence_data/iq/000001.mat … 001000.mat`.
- Annotation schema (per entry, parallel arrays over signals):
  `center_frequency` (Hz, in [−fs/2, fs/2]), `bandwidth` (Hz), `snr`, `modulation`,
  `channel`, `sample_rate` (=150000), `sample_num` (=12000; **stale** — the stored
  frames are 1200 samples), `sample_per_symbol`, `file_name`.
- `.mat` content: `signal_data` of shape `(num_signals, 2, 1200)` float64 — the
  **per-signal** passband I/Q components; the received frame is their **sum** over
  axis 0 (this is exactly what the historical `LoadFFTofCSRR` did). Keeping the
  components separate also gives us clean per-signal crops for classifier training.
- Signals per entry: 2–5 (3 and 4 dominate); modulations uniformly distributed
  over the 5 classes. Verified numerically that
  `bin = (cf/fs + 0.5)·1200` boxes capture the expected FFT energy
  (~⅓ of total energy per signal for a 3-signal entry) and that per-component FFT
  peaks fall inside their annotated bands.
- **No train/val/test annotation files exist on disk** — only raw per-entry JSONs.
  The new dataset classes therefore perform a deterministic seeded 50/10/40 split
  (repo convention, cf. `tools/convert_datasets/convert_amc.py`) over entries at
  load time.

Bandwidth in bins clusters around ~96/120/146 → matches the paper's
small/medium/large thresholds (110/130 bins) and motivates anchor widths
(100, 120, 140).

## 4. Design decisions for the re-implementation

1. **No mmdet/mmcv dependency** — 1-D IoU + 1-D NMS are ~40 lines of pure torch
   (`csrr/models/utils/interval_ops.py`). Boxes are `(left, right)` FFT-bin
   intervals; the time axis is by construction 100 % overlapped and never enters
   the IoU.
2. **Consistent grid geometry**: same-padding Conv1d backbone with three
   MaxPool(2) stages → stride 8, grid 150 for L=1200 (the old code's
   valid-padding trimming produced the 67-vs-144 inconsistency; the paper's own
   "144 units" is an artifact of valid padding). Anchor widths (100, 120, 140)
   bins, 3 anchors/cell per the paper.
3. **YOLOv3-style assignment done correctly**: responsible cell = cell containing
   the GT center; positive = best-IoU anchor in that cell; anchors with IoU >
   `ignore_iou_thr` (0.5) against any GT are excluded from the negative
   confidence loss; everything else is negative.
4. Losses as in the historical code / paper: BCE (confidence), BCE (within-cell
   center offset), MSE (log-bandwidth, weight 2.0), averaged over positives
   (confidence over positives+negatives).
5. Classifier trained on **ground-truth proposals** (baseband crops derived from
   the annotation), as in the paper's separate-training protocol; at inference the
   joint `JDMFramework` chains detector proposals into the classifier using an
   ideal (FFT-mask) low-pass instead of a FIR — equivalent brick-wall filtering,
   simpler and deterministic.
6. Metrics: pure-numpy COCO-style evaluator over 1-D IoU (mAP@[.5:.95], AP50,
   AP75, size-binned AP, AR, size-binned AR), registered as the mmengine metric
   `SignalDetectionMetric`; joint accuracy = same metric with `classwise=True` fed
   by the framework's per-detection modulation labels; plain `Accuracy` for the
   stand-alone classifier.
7. Splits: 50/10/40 (repo convention) with seed 0 over each version's 1000
   entries; configs default to all 124 versions (the paper also trains on the
   mixed dataset and evaluates per-condition).
