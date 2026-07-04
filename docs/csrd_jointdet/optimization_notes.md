# JDM Optimization Notes

Date: 2026-07-04

## Baseline

Dataset: `/home/citybuster/Data/WirelessRadio/data/ChangShuoTwc2026`

Detector checkpoint:
`work_dirs/jdm/jdm-det_fft-csrd/best_detection_mAP_epoch_2.pth`

Detector-only test metrics:

- mAP 0.6417, AP50 0.9893, AP75 0.5527, AR 0.7657
- SNR curve: `work_dirs/jdm/jdm-det_fft-csrd/snr_curve.json`
- Localization diagnostics:
  `work_dirs/jdm/diagnostics/det_baseline_test/test_localization.json`
  and `test_localization.csv`

Joint checkpoint: `work_dirs/jdm/jdm-joint_iq-csrd/jdm_joint.pth`

Joint/class-aware test metrics from the completed non-SNR run:

- mAP 0.3946, AP50 0.5236, AP75 0.4325, AR 0.5794

The later joint SNR run reached the end of inference but did not write final
metrics before becoming an orphaned process, so it should not be used as a
completed result.

## Detector Diagnosis

The AP50/AP75 gap is primarily a localization-quality problem, not proposal
recall:

- Best-overlap recall on the test split is 0.9986 at IoU50 but only 0.7403 at
  IoU75.
- Median center error is 1.02 FFT bins overall, so most boxes are centered well.
- Median bandwidth absolute error is 20.10 FFT bins overall, with many GTs in
  the 16-64 bin error range.
- Widths in the regenerated data are tightly clustered near 96, 120, and 146
  FFT bins. Current anchors (100, 120, 140) have mean center-aligned best IoU
  0.9748 on the test split; empirical anchors (96, 120, 146) improve this to
  0.9928.

By GT bandwidth bucket:

- Small: IoU75 best-overlap recall 0.9347, median center error 0.34 bins,
  median width error 3.74 bins.
- Medium: IoU75 best-overlap recall 0.9962, median center error 0.88 bins,
  median width error 22.06 bins.
- Large: IoU75 best-overlap recall 0.2365, median center error 18.56 bins,
  median width error 41.19 bins, mean signed width error -31.03 bins.

Interpretation:

- The grid/stride math is internally consistent: 1200 bins, stride 8, 150 cells,
  and continuous sigmoid center decode.
- Anchor quantization contributes but is too small to explain the large-box
  failure by itself.
- The dominant detector issue is wrong-width anchor selection/suppression for
  wider boxes. Objectness scores saturate near 1.0, so NMS can keep a
  medium/small-width anchor and suppress a better large-width anchor at the same
  center.
- Training logs also show `loss_bw` around 0.0001 late in training while center
  BCE remains around 0.5 due to continuous-label BCE entropy. Stronger bandwidth
  supervision is a reasonable first training variant.

## Variants Prepared

`configs/jdm/experiments/jdm-det_fft-csrd_anchor096146_bw20_5ep.py`

- 5-epoch bounded detector experiment.
- Uses empirical anchors `(96, 120, 146)`.
- Increases log-bandwidth MSE weight from 2 to 20.

`configs/jdm/experiments/jdm-det_fft-csrd_anchor096146_bw20.py`

- Full 30-epoch version of the same detector variant.
- Run only if the 5-epoch trend improves validation AP75/mAP.

`configs/jdm/experiments/jdm-det_fft-csrd_nms085_top6.py`

- Inference sensitivity config for anchor suppression.
- Uses empirical anchors and NMS IoU 0.85 with `max_per_frame=6` to keep metric
  aggregation bounded.

## Run Status

No new detector training was launched because GPU1 is occupied by the existing
JDM joint eval process and GPU0 is occupied by the AMR benchmark job. Do not
kill either process.

Recommended next commands when GPU1 is clear:

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
  configs/jdm/experiments/jdm-det_fft-csrd_anchor096146_bw20_5ep.py \
  --work-dir work_dirs/jdm/exp_det_anchor096146_bw20_5ep

CUDA_VISIBLE_DEVICES=1 python tools/test_det.py \
  configs/jdm/experiments/jdm-det_fft-csrd_nms085_top6.py \
  work_dirs/jdm/jdm-det_fft-csrd/best_detection_mAP_epoch_2.pth \
  --work-dir work_dirs/jdm/exp_det_anchor096146_nms085_top6
```

If the 5-epoch run improves validation AP75/mAP, run the full 30-epoch variant,
then rerun detector diagnostics and merge the resulting detector with the
existing AMC checkpoint for joint evaluation.
