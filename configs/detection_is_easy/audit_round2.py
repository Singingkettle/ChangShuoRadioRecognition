# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0.
"""Evidence-first audit utilities for the DetectionIsEasy paper.

The tool has no project-specific absolute paths.  Private locations are accepted
only as CLI arguments and may be recorded in the internal output manifest.

Examples
--------
Derive the paper-facing tables and statistical diagnostics::

    python configs/detection_is_easy/audit_round2.py derive \
      --taxonomy <cross-detector-taxonomy.csv> \
      --server-fetch <server-fetch.csv> --iou <iou_decile_data.csv> \
      --snr <snr_data.csv> --box-injection <box_injection_data.csv> \
      --quality-jsonl <box_quality_oracle_rcpA.jsonl> \
      --output-dir <internal-audit-dir> --paper-generated-dir <paper>/generated

Verify one or more prediction dumps with the same COCO evaluator::

    python configs/detection_is_easy/audit_round2.py same-pred \
      --annotation <instances_test.json> --prediction 20262811=<predictions.json> \
      --output <same-prediction-ap.json>

Verify the work-directory provenance after copying this script to the server::

    python audit_round2.py verify-server --taxonomy <taxonomy.csv> \
      --server-root <repository-root> --output <server-verification.json>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Iterable


ROUND_TOLERANCE = 5.00001e-5
EXPECTED_SEEDS = (101, 202, 303)
EXPECTED_FAMILIES = 13
FAMILY_DISPLAY = {
    "fcos": "FCOS",
    "sparse": "Sparse R-CNN",
    "atss": "ATSS",
    "dabdetr": "DAB-DETR",
    "deformable": "Deformable-DETR",
    "conddetr": "Conditional-DETR",
    "cascade": "Cascade R-CNN",
    "dino": "DINO",
    "gfl": "GFL",
    "yolox": "YOLOX",
    "faster": "Faster R-CNN",
    "retinanet": "RetinaNet",
    "rtmdet": "RTMDet",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_comment_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        lines = [line for line in stream if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"no CSV records in {path}")
    return list(csv.DictReader(lines))


def numeric(value: str | None) -> float | None:
    if value is None or value.strip().lower() in {"", "na", "n/a", "--", "n/r"}:
        return None
    return float(value)


def _sample_sd(values: Iterable[float]) -> float:
    vals = list(values)
    if len(vals) < 2:
        return float("nan")
    return statistics.stdev(vals)


def taxonomy_report(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_comment_csv(path)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["family"].strip()].append(row)
    errors: list[str] = []
    if len(rows) != EXPECTED_FAMILIES * len(EXPECTED_SEEDS):
        errors.append(f"expected 39 rows, found {len(rows)}")
    if len(grouped) != EXPECTED_FAMILIES:
        errors.append(f"expected 13 families, found {len(grouped)}")

    summary: list[dict[str, Any]] = []
    for family, family_rows in grouped.items():
        seeds = tuple(sorted(int(r["seed"]) for r in family_rows))
        if seeds != EXPECTED_SEEDS:
            errors.append(f"{family}: expected seeds {EXPECTED_SEEDS}, found {seeds}")
        deltas = [float(r["fused_delta"]) for r in family_rows]
        mean = statistics.fmean(deltas)
        sd = _sample_sd(deltas)
        claimed_means = {float(r["mean"]) for r in family_rows}
        claimed_sds = {float(r["sd"]) for r in family_rows}
        if len(claimed_means) != 1 or abs(next(iter(claimed_means)) - mean) > ROUND_TOLERANCE:
            errors.append(f"{family}: claimed mean {claimed_means} != {mean:.8f}")
        if len(claimed_sds) != 1 or abs(next(iter(claimed_sds)) - sd) > ROUND_TOLERANCE:
            errors.append(f"{family}: claimed sd {claimed_sds} != {sd:.8f}")
        det_values = {numeric(r.get("det_mAP")) for r in family_rows}
        if len(det_values) != 1:
            errors.append(f"{family}: inconsistent det_mAP values {det_values}")
        summary.append({
            "family": family,
            "seeds": list(seeds),
            "fused_delta_values": deltas,
            "fused_delta_mean": mean,
            "fused_delta_sd": sd,
            "det_mAP": next(iter(det_values)) if len(det_values) == 1 else None,
            "converged_lr": family_rows[0].get("converged_lr"),
            "rcpA_delta": numeric(family_rows[0].get("rcpA_delta")),
            "transfer_delta": numeric(family_rows[0].get("transfer_delta")),
        })
    summary.sort(key=lambda item: item["fused_delta_mean"], reverse=True)

    paired = [(r["det_mAP"], r["fused_delta_mean"]) for r in summary if r["det_mAP"] is not None]
    correlation: dict[str, Any] = {"n": len(paired)}
    if len(paired) >= 3:
        try:
            from scipy import stats

            x, y = zip(*paired)
            if len(set(x)) < 2 or len(set(y)) < 2:
                correlation.update({"pearson_r": None, "pearson_p_two_sided": None,
                                    "spearman_rho": None, "spearman_p_two_sided": None,
                                    "status": "NOT DEFINED: constant input"})
                return summary, {"errors": errors, "correlation": correlation, "rows": len(rows)}
            pearson = stats.pearsonr(x, y)
            spearman = stats.spearmanr(x, y)
            def finite_or_none(value: Any) -> float | None:
                number = float(value)
                return number if math.isfinite(number) else None

            correlation.update({
                "pearson_r": finite_or_none(pearson.statistic),
                "pearson_p_two_sided": finite_or_none(pearson.pvalue),
                "spearman_rho": finite_or_none(spearman.statistic),
                "spearman_p_two_sided": finite_or_none(spearman.pvalue),
            })
            pearson_r = float(pearson.statistic)
            z = math.atanh(max(-0.999999999, min(0.999999999, pearson_r)))
            half_width = 1.959963984540054 / math.sqrt(len(paired) - 3)
            correlation["pearson_ci_95_fisher"] = [
                math.tanh(z - half_width), math.tanh(z + half_width)]
            try:
                import numpy as np

                x_arr = np.asarray(x, dtype=float)
                y_arr = np.asarray(y, dtype=float)
                rng = np.random.default_rng(20260821)
                boot = []
                for _ in range(10_000):
                    indices = rng.integers(0, len(paired), size=len(paired))
                    bx, by = x_arr[indices], y_arr[indices]
                    if np.unique(bx).size < 2 or np.unique(by).size < 2:
                        continue
                    boot.append(float(stats.spearmanr(bx, by).statistic))
                correlation["spearman_ci_95_pairs_bootstrap"] = [
                    float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
                correlation["spearman_bootstrap_seed"] = 20260821
                correlation["spearman_bootstrap_requested"] = 10_000
                correlation["spearman_bootstrap_valid"] = len(boot)
            except ImportError:
                correlation["spearman_ci_95_pairs_bootstrap"] = None
        except ImportError:
            correlation["status"] = "NOT RUN: scipy unavailable"
    return summary, {"errors": errors, "correlation": correlation, "rows": len(rows)}


def iou_report(path: Path) -> dict[str, Any]:
    rows = read_comment_csv(path)
    fields = ("recog_acc", "acc_constellation", "acc_spectral")
    out: dict[str, Any] = {"n_deciles": len(rows), "series": {}}
    for field in fields:
        values = [float(r[field]) for r in rows]
        nondecreasing = all(a <= b for a, b in zip(values, values[1:]))
        nonincreasing = all(a >= b for a, b in zip(values, values[1:]))
        item: dict[str, Any] = {
            "values": values,
            "nondecreasing": nondecreasing,
            "nonincreasing": nonincreasing,
            "monotone": nondecreasing or nonincreasing,
        }
        try:
            from scipy import stats

            result = stats.spearmanr(range(len(values)), values)
            item.update({"spearman_rho": float(result.statistic),
                         "spearman_p_two_sided": float(result.pvalue)})
        except ImportError:
            item["statistics"] = "NOT RUN: scipy unavailable"
        out["series"][field] = item
    return out


def snr_report(path: Path) -> dict[str, Any]:
    rows = read_comment_csv(path)
    counts = [int(r["n_gt"]) for r in rows]
    return {
        "covered_count": sum(counts),
        "bucket_count": len(rows),
        "covered_lo_db": min(float(r["blocksnr_lo"]) for r in rows),
        "covered_hi_db": max(float(r["blocksnr_hi"]) for r in rows),
        "total_count": None,
        "excluded_count": None,
        "status": "PARTIAL: the CSV does not record the total number of signals outside its bins",
    }


def quality_report(path: Path) -> dict[str, Any]:
    ratios: list[float] = []
    cf_abs: list[float] = []
    scene_ids: set[str] = set()
    snr_values: list[float] = []
    n = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            n += 1
            if row.get("sid") is not None:
                scene_ids.add(str(row["sid"]))
            if row.get("bw_ratio") is not None:
                ratios.append(float(row["bw_ratio"]))
            if row.get("cf_err_bins_abs") is not None:
                cf_abs.append(float(row["cf_err_bins_abs"]))
            if row.get("snr_db") is not None:
                snr_values.append(float(row["snr_db"]))

    def q(values: list[float], quantiles: tuple[float, ...]) -> dict[str, float]:
        if not values:
            return {}
        try:
            import numpy as np

            arr = np.asarray(values, dtype=float)
            return {f"q{int(frac * 100):02d}": float(np.quantile(arr, frac)) for frac in quantiles}
        except ImportError:
            vals = sorted(values)
            return {f"q{int(frac * 100):02d}": vals[round(frac * (len(vals) - 1))]
                    for frac in quantiles}

    return {
        "records": n,
        "scenes": len(scene_ids),
        "bw_ratio_quantiles": q(ratios, (0.01, 0.1, 0.5, 0.9, 0.99)),
        "cf_error_bins_abs_quantiles": q(cf_abs, (0.01, 0.1, 0.5, 0.9, 0.99)),
        "metadata_snr_db_quantiles": q(snr_values, (0.01, 0.1, 0.5, 0.9, 0.99)),
    }


def box_injection_report(path: Path) -> dict[str, Any]:
    rows = read_comment_csv(path)
    baseline = next((float(r["overall_mAP"]) for r in rows
                     if r["perturbation"] == "cf_bins" and float(r["level"]) == 0.0), None)
    items = []
    for row in rows:
        value = float(row["overall_mAP"])
        items.append({
            "perturbation": row["perturbation"],
            "level": float(row["level"]),
            "overall_mAP": value,
            "delta_from_baseline": None if baseline is None else value - baseline,
            "source_log": row["source_log"],
        })
    return {"baseline": baseline, "rows": items,
            "status": "EXPLORATORY: perturbation levels are hand selected, not an empirical joint distribution"}


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def evidence_status_rows(same_pred_path: str | None) -> list[tuple[str, str]]:
    """Rows of the evidence-status table, read from a same-prediction result file.

    Without a file the rows state that nothing was run; with a file produced by
    ``same_pred_bootstrap.py`` (or ``audit_round2.py same-pred``) the rows report the
    number of detector seeds and whether the scene-paired bootstrap was run.  Nothing
    here is hard-coded to a hoped-for outcome.
    """
    if not same_pred_path:
        return [("Same model, predictions, split, and evaluator", "NOT RUN"),
                ("Three detector seeds", "NOT RUN"),
                ("Paired 2,000-resample interval", "NOT RUN")]
    data = json.loads(Path(same_pred_path).read_text(encoding="utf-8"))
    entries = data.get("predictions", [])
    n = len(entries)
    all_positive = all(float(e.get("delta_AP_loc_minus_AP_cls", -1)) > 0 for e in entries)
    boot = data.get("bootstrap")
    boot_done = isinstance(boot, dict) and all("bootstrap" in e for e in entries)
    resamples = boot.get("resamples") if isinstance(boot, dict) else None
    seeds_status = (f"PASS ({n} seeds, all deltas positive)" if n >= 3 and all_positive
                    else f"OPEN ({n} of 3 dumps)" if n < 3 else f"FAIL ({n} seeds, a delta is not positive)")
    return [("Same model, predictions, split, and evaluator", f"PASS ({n} seed{'s' if n != 1 else ''})"),
            ("Three detector seeds", seeds_status),
            (f"Paired {resamples or '2,000'}-resample interval",
             "DONE (scene-paired)" if boot_done else "NOT RUN")]


def derive(args: argparse.Namespace) -> int:
    inputs = {
        "taxonomy": Path(args.taxonomy).resolve(),
        "server_fetch": Path(args.server_fetch).resolve(),
        "iou": Path(args.iou).resolve(),
        "snr": Path(args.snr).resolve(),
        "box_injection": Path(args.box_injection).resolve(),
    }
    if args.quality_jsonl:
        inputs["quality_jsonl"] = Path(args.quality_jsonl).resolve()
    for name, path in inputs.items():
        if not path.is_file():
            raise SystemExit(f"missing {name}: {path}")

    taxonomy, taxonomy_meta = taxonomy_report(inputs["taxonomy"])
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "statistics": {"sd": "sample standard deviation (ddof=1)",
                       "correlations": "two-sided", "rounding": "three decimals in manuscript"},
        "taxonomy": taxonomy_meta,
        "iou_deciles": iou_report(inputs["iou"]),
        "snr": snr_report(inputs["snr"]),
        "box_injection": box_injection_report(inputs["box_injection"]),
    }
    if "quality_jsonl" in inputs:
        report["box_quality"] = quality_report(inputs["quality_jsonl"])

    output_dir = Path(args.output_dir).resolve()
    generated_dir = Path(args.paper_generated_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    taxonomy_csv_rows = [{
        "family": row["family"],
        "seed_101": row["fused_delta_values"][0],
        "seed_202": row["fused_delta_values"][1],
        "seed_303": row["fused_delta_values"][2],
        "mean": f'{row["fused_delta_mean"]:.8f}',
        "sample_sd": f'{row["fused_delta_sd"]:.8f}',
        "det_mAP": "na" if row["det_mAP"] is None else row["det_mAP"],
        "converged_lr": row["converged_lr"],
    } for row in taxonomy]
    fields = ["family", "seed_101", "seed_202", "seed_303", "mean", "sample_sd",
              "det_mAP", "converged_lr"]
    _write_csv(output_dir / "round2-paper-values.csv", taxonomy_csv_rows, fields)

    report_path = generated_dir / "statistical_report.json"
    _write_text(report_path, json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    _write_text(output_dir / "statistical_report.json",
                json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    corr = taxonomy_meta["correlation"]
    def tex_number(value: Any) -> str:
        return "na" if value is None else f"{float(value):.3f}"

    macros = [
        "% Generated by configs/detection_is_easy/audit_round2.py; do not edit.",
        f"% taxonomy_sha256={sha256(inputs['taxonomy'])}",
        f"\\newcommand{{\\TaxonomyFamilies}}{{{len(taxonomy)}}}",
        f"\\newcommand{{\\TaxonomyRows}}{{{sum(len(r['fused_delta_values']) for r in taxonomy)}}}",
        f"\\newcommand{{\\TaxonomyPearson}}{{{tex_number(corr.get('pearson_r'))}}}",
        f"\\newcommand{{\\TaxonomyPearsonP}}{{{tex_number(corr.get('pearson_p_two_sided'))}}}",
        f"\\newcommand{{\\TaxonomySpearman}}{{{tex_number(corr.get('spearman_rho'))}}}",
        f"\\newcommand{{\\TaxonomySpearmanP}}{{{tex_number(corr.get('spearman_p_two_sided'))}}}",
        f"\\newcommand{{\\TaxonomyPearsonCILo}}{{{tex_number((corr.get('pearson_ci_95_fisher') or [None])[0])}}}",
        f"\\newcommand{{\\TaxonomyPearsonCIHi}}{{{tex_number((corr.get('pearson_ci_95_fisher') or [None, None])[1])}}}",
        f"\\newcommand{{\\TaxonomySpearmanCILo}}{{{tex_number((corr.get('spearman_ci_95_pairs_bootstrap') or [None])[0])}}}",
        f"\\newcommand{{\\TaxonomySpearmanCIHi}}{{{tex_number((corr.get('spearman_ci_95_pairs_bootstrap') or [None, None])[1])}}}",
        f"\\newcommand{{\\SNRCountInBins}}{{{report['snr']['covered_count']:,}}}",
    ]
    _write_text(generated_dir / "paper_macros.tex", "\n".join(macros) + "\n")

    status_rows = evidence_status_rows(getattr(args, "same_pred", None))
    core = [
        "% Generated evidence status; safe to include in a draft.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Evidence status for the central same-prediction comparison.}",
        "\\label{tab:evidence-status}",
        "\\footnotesize",
        "\\begin{tabular}{p{0.58\\columnwidth}p{0.32\\columnwidth}}",
        "\\toprule",
        "Requirement & Status \\\\",
        "\\midrule",
        *[f"{name} & {status} \\\\" for name, status in status_rows],
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    _write_text(generated_dir / "core_results.tex", "\n".join(core) + "\n")

    tax_lines = [
        "% Generated taxonomy table from the 39 seed rows.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Matched predicted-box training across detector families. Values are three-seed mean $\\pm$ sample standard deviation.}",
        "\\label{tab:taxonomy-generated}",
        f"{chr(92)}footnotesize{chr(92)}setlength{{{chr(92)}tabcolsep}}{{4pt}}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Detector & Detector AP & $\\Delta$ operational AP \\\\",
        "\\midrule",
    ]
    for row in taxonomy:
        det = "na" if row["det_mAP"] is None else f'{row["det_mAP"]:.3f}'
        name = FAMILY_DISPLAY.get(row["family"], row["family"]).replace("_", "\\_")
        tax_lines.append(f"{name} & {det} & {row['fused_delta_mean']:+.3f} $\\pm$ {row['fused_delta_sd']:.3f} \\\\")
    tax_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    _write_text(generated_dir / "taxonomy_summary.tex", "\n".join(tax_lines) + "\n")

    detail_lines = [
        "% Generated seed-level taxonomy values from the immutable campaign CSV.",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Seed-level matched predicted-box gains. The last column uses sample standard deviation ($\\mathrm{ddof}=1$).}",
        "\\label{tab:taxonomy-seeds}",
        "\\scriptsize\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Detector & Detector AP & Seed 101 & Seed 202 & Seed 303 & Mean $\\pm$ SD \\\\",
        "\\midrule",
    ]
    for row in taxonomy:
        det = "na" if row["det_mAP"] is None else f'{row["det_mAP"]:.3f}'
        name = FAMILY_DISPLAY.get(row["family"], row["family"]).replace("_", "\\_")
        seeds = row["fused_delta_values"]
        detail_lines.append(
            f"{name} & {det} & {seeds[0]:+.5f} & {seeds[1]:+.5f} & "
            f"{seeds[2]:+.5f} & {row['fused_delta_mean']:+.3f} $\\pm$ "
            f"{row['fused_delta_sd']:.3f} \\\\")
    detail_lines.extend(["\\bottomrule", "\\end{tabular}}", "\\end{table}"])
    _write_text(generated_dir / "taxonomy_seed_details.tex",
                "\n".join(detail_lines) + "\n")

    evidence = {
        "schema_version": 1,
        "generated_at_utc": report["generated_at_utc"],
        "inputs": {name: {"path": str(path), "bytes": path.stat().st_size,
                          "sha256": sha256(path)} for name, path in inputs.items()},
        "outputs": {},
        "status": "FAIL" if taxonomy_meta["errors"] else "PASS",
        "limitations": [
            "server provenance is a separate gate",
            "same-prediction AP requires prediction dumps and COCO annotations",
            "the SNR CSV does not expose the number of signals outside the plotted bins",
        ],
    }
    for path in sorted(output_dir.glob("*")):
        if path.is_file() and path.name != "round2-evidence-manifest.json":
            evidence["outputs"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    _write_text(output_dir / "round2-evidence-manifest.json",
                json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": evidence["status"], "taxonomy_errors": taxonomy_meta["errors"],
                      "output_dir": str(output_dir)}, indent=2))
    return 1 if taxonomy_meta["errors"] else 0


def evaluate_same_predictions(annotation: Path, prediction: Path) -> dict[str, float]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise SystemExit("pycocotools is required for same-pred evaluation") from exc

    sink = StringIO()
    with redirect_stdout(sink):
        gt = COCO(str(annotation))
        dt = gt.loadRes(str(prediction))
    result: dict[str, float] = {}
    for label, use_categories in (("AP_cls", 1), ("AP_loc", 0)):
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.params.useCats = use_categories
        evaluator.params.imgIds = sorted(gt.getImgIds())
        with redirect_stdout(sink):
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
        result[label] = float(evaluator.stats[0])
    result["delta_AP_loc_minus_AP_cls"] = result["AP_loc"] - result["AP_cls"]
    return result


def same_pred(args: argparse.Namespace) -> int:
    annotation = Path(args.annotation).resolve()
    entries: list[dict[str, Any]] = []
    for item in args.prediction:
        if "=" not in item:
            raise SystemExit("--prediction must be SEED=PATH")
        seed, raw_path = item.split("=", 1)
        path = Path(raw_path).resolve()
        entries.append({"seed": seed, "path": str(path), "sha256": sha256(path),
                        **evaluate_same_predictions(annotation, path)})
    deltas = [row["delta_AP_loc_minus_AP_cls"] for row in entries]
    gate = (len(entries) == 3 and all(delta > 0 for delta in deltas)
            and statistics.fmean(deltas) >= 0.10)
    output = {
        "schema_version": 1,
        "annotation": {"path": str(annotation), "sha256": sha256(annotation)},
        "predictions": entries,
        "metric": "COCO bbox AP@[.50:.95], identical prediction records; useCats=0 vs useCats=1",
        "bootstrap": "NOT RUN: requires all three preregistered seed dumps",
        "strong_title_gate": "INCOMPLETE" if len(entries) != 3 else ("PASS" if gate else "FAIL"),
        "required_title": ("Localization Is Easier Than Recognition: A Controlled Study of Wideband RF Signal Analysis"
                           if gate else "Disentangling Localization and Recognition in Wideband RF Signal Analysis"),
    }
    path = Path(args.output).resolve()
    _write_text(path, json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0 if len(entries) == 3 else 2


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _summary_metrics(path: Path) -> dict[str, float]:
    rows = read_comment_csv(path)
    metrics: dict[str, float] = {}
    for row in rows:
        keys = {key.lower(): value for key, value in row.items() if key is not None}
        metric = keys.get("metric") or keys.get("name") or keys.get("key")
        value = keys.get("value")
        if metric is not None and value is not None:
            try:
                metrics[metric] = float(value)
            except ValueError:
                pass
        for key, raw in row.items():
            if raw is None:
                continue
            try:
                metrics.setdefault(key, float(raw))
            except (TypeError, ValueError):
                pass
    return metrics


def _summary_table(path: Path) -> dict[tuple[str, str], float]:
    table: dict[tuple[str, str], float] = {}
    for row in read_comment_csv(path):
        split = (row.get("split") or "").strip()
        metric = (row.get("metric") or "").strip()
        value = row.get("value")
        if split and metric and value is not None:
            try:
                table[(split, metric)] = float(value)
            except ValueError:
                continue
    return table


def verify_server(args: argparse.Namespace) -> int:
    taxonomy = Path(args.taxonomy).resolve()
    root = Path(args.server_root).resolve()
    rows = read_comment_csv(taxonomy)
    checks: list[dict[str, Any]] = []
    detector_checks: list[dict[str, Any]] = []
    failures = 0
    for row in rows:
        relative = Path(row["matched_work_dir"]) / "summary.csv"
        target = (root / relative).resolve()
        item: dict[str, Any] = {"family": row["family"], "seed": int(row["seed"]),
                                "relative_path": relative.as_posix(), "exists": target.is_file()}
        if not _within(target, root):
            item["error"] = "path escapes server root"
            failures += 1
        elif not target.is_file():
            item["error"] = "summary.csv missing"
            failures += 1
        else:
            text = target.read_text(encoding="utf-8", errors="replace")
            expected = float(row["fused_delta"])
            token = None
            for line in text.splitlines():
                if "fused_delta" in line:
                    cells = [cell.strip() for cell in line.split(",")]
                    for cell in reversed(cells):
                        try:
                            token = float(cell)
                            break
                        except ValueError:
                            continue
                if token is not None:
                    break
            item.update({"sha256": sha256(target), "expected_fused_delta": expected,
                         "observed_fused_delta": token})
            if token is None or abs(token - expected) > ROUND_TOLERANCE:
                item["error"] = "fused_delta missing or inconsistent"
                failures += 1
        checks.append(item)
        det_relative_raw = (row.get("det_work_dir") or "").strip()
        det_value = numeric(row.get("det_mAP"))
        if det_relative_raw.lower() in {"", "na", "n/a"}:
            det_item = {"family": row["family"], "seed": int(row["seed"]),
                        "relative_path": "na", "expected_det_mAP": det_value,
                        "status": "NA WITH REASON IN TAXONOMY"}
            if det_value is not None:
                det_item["error"] = "det_mAP is numeric while det_work_dir is na"
                failures += 1
        else:
            det_target = (root / Path(det_relative_raw) / "summary.csv").resolve()
            det_item = {"family": row["family"], "seed": int(row["seed"]),
                        "relative_path": (Path(det_relative_raw) / "summary.csv").as_posix(),
                        "expected_det_mAP": det_value, "exists": det_target.is_file()}
            if not _within(det_target, root) or not det_target.is_file():
                det_item["error"] = "detector summary missing or escapes server root"
                failures += 1
            else:
                observed = _summary_table(det_target).get(("val", "bbox_mAP"))
                det_item.update({"observed_det_mAP": observed, "sha256": sha256(det_target)})
                if det_value is None or observed is None or abs(det_value - observed) > ROUND_TOLERANCE:
                    det_item["error"] = "det_mAP missing or inconsistent"
                    failures += 1
        detector_checks.append(det_item)

    fetch_checks: list[dict[str, Any]] = []
    if args.server_fetch:
        for row in read_comment_csv(Path(args.server_fetch).resolve()):
            relative = Path(row["work_dir"]) / "summary.csv"
            target = (root / relative).resolve()
            split, metric = [part.strip() for part in row["metric"].split(",", 1)]
            expected = float(row["value"])
            item = {"kind": row["kind"], "family": row["family"], "seed": int(row["seed"]),
                    "relative_path": relative.as_posix(), "split": split, "metric": metric,
                    "expected": expected, "exists": target.is_file()}
            if not _within(target, root) or not target.is_file():
                item["error"] = "server-fetch summary missing or escapes server root"
                failures += 1
            else:
                observed = _summary_table(target).get((split, metric))
                item.update({"observed": observed, "sha256": sha256(target)})
                if observed is None or abs(expected - observed) > ROUND_TOLERANCE:
                    item["error"] = "server-fetch metric missing or inconsistent"
                    failures += 1
            fetch_checks.append(item)

    total_checks = len(checks) + len(detector_checks) + len(fetch_checks)
    result = {"schema_version": 1, "server_root": str(root), "matched_checks": checks,
              "detector_checks": detector_checks, "server_fetch_checks": fetch_checks,
              "verified": total_checks - failures, "failed": failures,
              "status": "PASS" if failures == 0 else "FAIL"}
    _write_text(Path(args.output).resolve(), json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": result["status"], "verified": result["verified"],
                      "failed": failures}, indent=2))
    return 1 if failures else 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("derive")
    p.add_argument("--taxonomy", required=True)
    p.add_argument("--server-fetch", required=True)
    p.add_argument("--iou", required=True)
    p.add_argument("--snr", required=True)
    p.add_argument("--box-injection", required=True)
    p.add_argument("--quality-jsonl")
    p.add_argument("--same-pred", help="same-prediction result JSON (same_pred_bootstrap.py or same-pred); "
                                       "renders the evidence-status rows from it")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--paper-generated-dir", required=True)
    p.set_defaults(func=derive)

    p = sub.add_parser("same-pred")
    p.add_argument("--annotation", required=True)
    p.add_argument("--prediction", action="append", required=True, metavar="SEED=PATH")
    p.add_argument("--output", required=True)
    p.set_defaults(func=same_pred)

    p = sub.add_parser("verify-server")
    p.add_argument("--taxonomy", required=True)
    p.add_argument("--server-fetch")
    p.add_argument("--server-root", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=verify_server)
    return ap


def main() -> int:
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
