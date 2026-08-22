import importlib.util
import json
from pathlib import Path

import pytest


ROOT = next(
    candidate for candidate in Path(__file__).resolve().parents
    if (candidate / "tools" / "misc" / "check_paper.py").exists()
)
SCRIPT = ROOT / "configs" / "detection_is_easy" / "audit_round2.py"
SPEC = importlib.util.spec_from_file_location("audit_round2", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def taxonomy(tmp_path: Path) -> Path:
    path = tmp_path / "taxonomy.csv"
    lines = ["# provenance comment", "family,seed,fused_delta,mean,sd,det_mAP,rcpA_delta,transfer_delta,converged_lr,matched_work_dir,det_work_dir,source"]
    for family_index in range(13):
        family = f"f{family_index}"
        values = (0.1 + family_index / 1000, 0.101 + family_index / 1000, 0.102 + family_index / 1000)
        mean = sum(values) / 3
        sd = 0.001
        for seed, value in zip((101, 202, 303), values):
            lines.append(f"{family},{seed},{value:.4f},{mean:.4f},{sd:.4f},0.4,0.01,0.02,5e-4,wd/{family}/{seed},det/{family},test")
    write(path, "\n".join(lines) + "\n")
    return path


def test_taxonomy_recomputes_all_families(tmp_path):
    summary, meta = AUDIT.taxonomy_report(taxonomy(tmp_path))
    assert len(summary) == 13
    assert meta["rows"] == 39
    assert meta["errors"] == []


def test_taxonomy_rejects_fabricated_summary(tmp_path):
    path = taxonomy(tmp_path)
    text = path.read_text(encoding="utf-8").replace("f0,101,0.1000,0.1010", "f0,101,0.1000,0.9000")
    write(path, text)
    _, meta = AUDIT.taxonomy_report(path)
    assert any("claimed mean" in error for error in meta["errors"])


def test_iou_report_does_not_invent_monotonicity(tmp_path):
    path = tmp_path / "iou.csv"
    write(path, "decile,recog_acc,acc_constellation,acc_spectral\n0,0.1,0.2,0.3\n1,0.3,0.1,0.4\n2,0.2,0.4,0.2\n")
    report = AUDIT.iou_report(path)
    assert not report["series"]["recog_acc"]["monotone"]
    assert not report["series"]["acc_constellation"]["monotone"]
    assert not report["series"]["acc_spectral"]["monotone"]


def test_server_path_boundary(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    assert AUDIT._within(root / "work_dirs" / "run", root)
    assert not AUDIT._within(root / ".." / "outside", root)


def test_server_verification_checks_matched_and_detector_summaries(tmp_path):
    tax = taxonomy(tmp_path)
    root = tmp_path / "repo"
    for family_index in range(13):
        family = f"f{family_index}"
        for seed, delta in zip((101, 202, 303),
                               (0.1 + family_index / 1000,
                                0.101 + family_index / 1000,
                                0.102 + family_index / 1000)):
            path = root / "wd" / family / str(seed) / "summary.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            write(path, f"split,metric,value\ntest,fused_delta,{delta:.4f}\n")
        det = root / "det" / family / "summary.csv"
        det.parent.mkdir(parents=True, exist_ok=True)
        write(det, "split,metric,value\nval,bbox_mAP,0.4\n")
    output = tmp_path / "verified.json"
    args = type("Args", (), {"taxonomy": str(tax), "server_fetch": None,
                              "server_root": str(root), "output": str(output)})
    assert AUDIT.verify_server(args) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "PASS"
    assert len(result["matched_checks"]) == 39
    assert len(result["detector_checks"]) == 39


def test_same_pred_requires_seed_path_separator(tmp_path):
    args = type("Args", (), {"annotation": "missing", "prediction": ["bad"], "output": str(tmp_path / "out.json")})
    with pytest.raises(SystemExit, match="SEED=PATH"):
        AUDIT.same_pred(args)


def test_evidence_json_has_no_nan_on_valid_taxonomy(tmp_path):
    summary, meta = AUDIT.taxonomy_report(taxonomy(tmp_path))
    encoded = json.dumps({"summary": summary, "meta": meta}, allow_nan=False)
    assert '"family": "f0"' in encoded


def test_evidence_status_rows_follow_the_result_file(tmp_path):
    rows = AUDIT.evidence_status_rows(None)
    assert all(status == "NOT RUN" for _, status in rows)
    one = tmp_path / "one.json"
    one.write_text(json.dumps({"predictions": [{"seed": "a", "delta_AP_loc_minus_AP_cls": 0.2}],
                               "bootstrap": "NOT RUN"}), encoding="utf-8")
    rows = dict(AUDIT.evidence_status_rows(str(one)))
    assert rows["Three detector seeds"].startswith("OPEN (1 of 3")
    assert rows["Paired 2,000-resample interval"] == "NOT RUN"
    three = tmp_path / "three.json"
    three.write_text(json.dumps({"predictions": [
        {"seed": s, "delta_AP_loc_minus_AP_cls": d, "bootstrap": {}} for s, d in (("a", 0.2), ("b", 0.1), ("c", -0.1))],
        "bootstrap": {"resamples": 50}}), encoding="utf-8")
    rows = dict(AUDIT.evidence_status_rows(str(three)))
    assert rows["Three detector seeds"].startswith("FAIL")
    assert rows["Paired 50-resample interval"] == "DONE (scene-paired)"
