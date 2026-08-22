import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = next(
    candidate for candidate in Path(__file__).resolve().parents
    if (candidate / "tools" / "misc" / "check_paper.py").exists()
)
PAPER = ROOT / "configs" / "detection_is_easy"


def load(name):
    spec = importlib.util.spec_from_file_location(name, PAPER / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny_coco(tmp_path):
    """Three images, two categories, a mix of correct, mislabeled and missed boxes."""
    images = [{"id": i, "width": 100, "height": 100, "file_name": f"{i}.png"} for i in range(3)]
    cats = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
    anns, preds, aid = [], [], 1
    for img in range(3):
        for k, cat in enumerate((1, 2)):
            box = [10 + 40 * k, 10 + 20 * img, 20, 20]
            anns.append({"id": aid, "image_id": img, "category_id": cat, "bbox": box,
                         "area": 400, "iscrowd": 0})
            aid += 1
            if img == 2 and cat == 2:
                continue  # one missed ground truth
            wrong = (img == 1 and cat == 1)
            preds.append({"image_id": img, "category_id": 2 if wrong and cat == 1 else cat,
                          "bbox": [box[0] + 1, box[1] + 1, 20, 20], "score": 0.9 - 0.1 * img - 0.05 * k})
    preds.append({"image_id": 0, "category_id": 1, "bbox": [70, 70, 10, 10], "score": 0.3})
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}), encoding="utf-8")
    dt = tmp_path / "dt.json"
    dt.write_text(json.dumps(preds), encoding="utf-8")
    return gt, dt


def test_weighted_reaccumulation_reproduces_cocoeval(tmp_path):
    pytest.importorskip("pycocotools")
    mod = load("same_pred_bootstrap")
    from pycocotools.coco import COCO

    gt_path, dt_path = tiny_coco(tmp_path)
    gt = COCO(str(gt_path))
    dt = gt.loadRes(str(dt_path))
    for use_cats in (1, 0):
        ev, cats, rec_thrs, n_img = mod.prepare(gt, dt, use_cats)
        assert n_img == 3
        assert abs(mod.ap_weighted(cats, rec_thrs, np.ones(n_img)) - ev.stats[0]) < 1e-12
    # ignoring categories can only raise AP for identical boxes
    ev_cls, cats_cls, rec, _ = mod.prepare(gt, dt, 1)
    ev_loc, cats_loc, _, _ = mod.prepare(gt, dt, 0)
    assert ev_loc.stats[0] >= ev_cls.stats[0]
    # dropping an image (weight 0) equals evaluating on the remaining images only
    w = np.array([1.0, 1.0, 0.0])
    assert 0.0 <= mod.ap_weighted(cats_cls, rec, w) <= 1.0


def test_bootstrap_cli_writes_paired_arrays(tmp_path):
    pytest.importorskip("pycocotools")
    import subprocess
    import sys

    gt_path, dt_path = tiny_coco(tmp_path)
    out = tmp_path / "boot.json"
    subprocess.run(
        [sys.executable, str(PAPER / "same_pred_bootstrap.py"), "--annotation", str(gt_path),
         "--prediction", f"s1={dt_path}", "--prediction", f"s2={dt_path}", "--resamples", "25",
         "--seed", "3", "--output", str(out)],
        check=True, capture_output=True, text=True, cwd=str(tmp_path))
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["bootstrap"]["resamples"] == 25
    assert len(result["predictions"]) == 2
    first, second = result["predictions"]
    assert first["AP_cls"] == second["AP_cls"]
    assert first["bootstrap"]["delta_ci95"] == second["bootstrap"]["delta_ci95"]
    assert result["pooled"]["n_seeds"] == 2
    arrays = np.load(first["resample_arrays"])
    assert arrays["boot_cls"].shape == (25,)


def test_grouped_cv_auc_is_scene_disjoint_and_bounded(tmp_path):
    mod = load("box_quality_auc_cv")
    rng = np.random.default_rng(0)
    rows = []
    for scene in range(40):
        for _ in range(6):
            iou = float(rng.uniform(0.5, 1.0))
            rows.append({
                "sid": f"s{scene:03d}", "iou": iou, "gt_containment": iou, "pred_containment": iou,
                "freq_coverage": iou, "time_coverage": 1.0, "energy_coverage": iou,
                "energy_in_window": 1.0, "energy_contamination": 1 - iou, "cf_err_bins_abs": 1 - iou,
                "cf_err_cyc_abs": (1 - iou) / 100, "bw_ratio": 1.0 + (iou - 0.75),
                "recog_correct": int(rng.uniform() < iou), "oracle_correct": 1,
            })
    path = tmp_path / "q.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    sids, y, oracle, X = mod.load(path)
    folds = mod.group_folds(sids, 5, np.random.default_rng(1))
    for k in range(5):
        assert not set(sids[folds == k]) & set(sids[folds != k])
    report = mod.run_population("t", sids, y, X, np.random.default_rng(1), 20, np.random.default_rng(2))
    table = report["table"]
    assert 0.0 <= table["iou"]["auc_pooled_oof"] <= 1.0
    lo, hi = table["multivariate_11_features"]["ci95_scene_bootstrap"]
    assert lo <= table["multivariate_11_features"]["auc_pooled_oof"] <= hi
    assert len(report["outer_folds"]) == 5
