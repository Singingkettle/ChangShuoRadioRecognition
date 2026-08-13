# AMR-Benchmark reproduction index

This folder is an **index** for the CSRR ports of models surveyed in
Zhang et al., "Deep Learning Based Automatic Modulation Recognition: Models,
Datasets, and Challenges," *Digital Signal Processing*, 2022, and the
[Richardzhangxx/AMR-Benchmark](https://github.com/Richardzhangxx/AMR-Benchmark)
Keras reference — plus CSRR own methods (MLDNN / FastMLDNN / HCGDNN).

**Algorithm short name = `configs/<name>/` = `docs/<name>/`.** Per-method
READMEs own the paper citation, train commands, and measured results. Do not
treat `amr_benchmark` as an algorithm name.

## Campaign status (closed 2026-08-14)

- Tracking matrix: **44 pass / 17 fail / 11 measured** under approximate
  tolerances (overall ≥ target−2.0 pp, peak ≥ target−1.5 pp with near-match).
- **Hisar split is already official** (Test + Train 80/20). Remaining Hisar
  fails are not a 50/10/40 partition bug.
- RML DeepSig stays CSRR **50/10/40** vs TF **6:2:2** (~2–4 pp Tier-B bias).
- Remaining fails are mostly RML2018.01A long-sequence / structural ceilings
  and Hisar read-off / hard cases (MCNET, DAE peak, CLDNN*). Further GPU
  siege (seed lottery, SelfNormalize FT that collapses val, FastMLDNN reuse
  spins) is **closed**.

## Method docs

| Method | Docs | Configs |
|---|---|---|
| CGDNet | [docs/cgdnet](../cgdnet) | [configs/cgdnet](../../configs/cgdnet) |
| CLDNNW | [docs/cldnnw](../cldnnw) | [configs/cldnnw](../../configs/cldnnw) |
| CLDNNL | [docs/cldnnl](../cldnnl) | [configs/cldnnl](../../configs/cldnnl) |
| CNN1DPF | [docs/cnn1dpf](../cnn1dpf) | [configs/cnn1dpf](../../configs/cnn1dpf) |
| CNN2 | [docs/cnn2](../cnn2) | [configs/cnn2](../../configs/cnn2) |
| CNN4 | [docs/cnn4](../cnn4) | [configs/cnn4](../../configs/cnn4) |
| DAE | [docs/dae](../dae) | [configs/dae](../../configs/dae) |
| DensCNN | [docs/denscnn](../denscnn) | [configs/denscnn](../../configs/denscnn) |
| DSCLDNN | [docs/dscldnn](../dscldnn) | [configs/dscldnn](../../configs/dscldnn) |
| FastMLDNN | [docs/fastmldnn](../fastmldnn) | [configs/fastmldnn](../../configs/fastmldnn) |
| GRU2 | [docs/gru2](../gru2) | [configs/gru2](../../configs/gru2) |
| HCGDNN | [docs/hcgdnn](../hcgdnn) | [configs/hcgdnn](../../configs/hcgdnn) |
| IC-AMCNet | [docs/icamcnet](../icamcnet) | [configs/icamcnet](../../configs/icamcnet) |
| LSTM2 | [docs/lstm2](../lstm2) | [configs/lstm2](../../configs/lstm2) |
| MCformer | [docs/mcformer](../mcformer) | [configs/mcformer](../../configs/mcformer) |
| MCLDNN | [docs/mcldnn](../mcldnn) | [configs/mcldnn](../../configs/mcldnn) |
| MCNET | [docs/mcnet](../mcnet) | [configs/mcnet](../../configs/mcnet) |
| MLDNN | [docs/mldnn](../mldnn) | [configs/mldnn](../../configs/mldnn) |
| PET-CGDNN | [docs/petcgdnn](../petcgdnn) | [configs/petcgdnn](../../configs/petcgdnn) |
| ResNetAMR | [docs/resnetamr](../resnetamr) | [configs/resnetamr](../../configs/resnetamr) |
| TRN | [docs/trn](../trn) | [configs/trn](../../configs/trn) |

Ingest rules for new papers: [`docs/adding_a_new_paper.md`](../adding_a_new_paper.md).
