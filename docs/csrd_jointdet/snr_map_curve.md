# JDM mAP vs SNR Curves

The JDM SNR curve is a per-object breakdown of the detection metric. Each CSRD
frame may contain multiple signals, and each annotation stores an `snr` array
parallel to `gt_boxes` and `gt_box_labels`. Therefore the SNR bin for a ground
truth interval is the SNR of that individual signal, not a frame-level value.

Frame-level grouping is wrong for this dataset because one received frame can
contain boxes with different SNRs. Assigning the entire frame to one SNR would
either duplicate or drop positives from the wrong bins and can make the curve
depend on which signal happened to be chosen as the frame label.

## Metric Definition

`SignalDetectionMetric(snrwise=True)` collects `snr` from the detection data
sample metainfo and validates that it has exactly one value per GT box. For an
SNR value `s`, it keeps only GT boxes whose own `snr == s` as positives.
Detections that overlap annotated boxes from other SNR bins are ignored for
that bin, so correctly detecting another signal in the same frame is not
counted as a false positive. The positive matches still use the same 1-D
interval IoU, greedy matching, COCO-style 101-point AP, class selection, and AR
code paths as the aggregate JDM metric.

For detector-only evaluation, predictions and GT are class agnostic. For joint
JDM evaluation, `classwise=True` first evaluates each modulation class and then
averages over classes, matching the aggregate class-aware joint mAP.

## Reproduce

The regenerated CSRD data root used by the JDM configs is:

```bash
/home/citybuster/Data/WirelessRadio/data/ChangShuoTwc2026
```

Detector-only test with SNR curve:

```bash
CUDA_VISIBLE_DEVICES=1 python tools/test_det.py \
    configs/jdm/jdm-det_fft-csrd.py \
    work_dirs/jdm/jdm-det_fft-csrd/best_detection_mAP_epoch_2.pth \
    --work-dir work_dirs/jdm/jdm-det_fft-csrd
```

Joint/class-aware test with SNR curve:

```bash
python tools/merge_jdm_checkpoints.py \
    work_dirs/jdm/jdm-det_fft-csrd/best_detection_mAP_epoch_2.pth \
    work_dirs/jdm/jdm-amc_iq-csrd/best_accuracy_top1_epoch_60.pth \
    work_dirs/jdm/jdm-joint_iq-csrd/jdm_joint.pth
CUDA_VISIBLE_DEVICES=1 python tools/test_det.py \
    configs/jdm/jdm-joint_iq-csrd.py \
    work_dirs/jdm/jdm-joint_iq-csrd/jdm_joint.pth \
    --work-dir work_dirs/jdm/jdm-joint_iq-csrd
```

When `snrwise=True`, `tools/test_det.py` writes the curve artifacts under the
active work directory by default. The detector config writes mAP and AR per
SNR; the joint config writes class-aware mAP per SNR.

```bash
work_dirs/jdm/jdm-det_fft-csrd/snr_curve.json
work_dirs/jdm/jdm-det_fft-csrd/snr_curve.pdf
work_dirs/jdm/jdm-joint_iq-csrd/snr_curve.json
work_dirs/jdm/jdm-joint_iq-csrd/snr_curve.pdf
```
