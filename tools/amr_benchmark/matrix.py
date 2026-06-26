"""Model × dataset matrix for the AMR-Benchmark reproduction sweep.

Phase 2 consumes this file via ``tools/amr_benchmark/run_migration.py`` to
schedule training/testing for every entry. Each entry pairs a CSRR config
with the reference accuracy targets documented in
``docs/amr_benchmark/accuracy_targets.md`` so the orchestrator can compute
pass/fail labels automatically.

Editing rules:

* ``model`` keys MUST match the CSRR backbone class name (lower-cased) and
  the directory under ``configs/`` (e.g. ``cnn2`` for ``CNN2``).
* ``dataset`` keys use short labels (``deepsig201610A``, ``deepsig201610B``,
  ``deepsig201801A``, ``hisar2019``). The orchestrator's ``--dataset`` flag
  accepts the same labels.
* ``config`` is the path (relative to repository root) of the CSRR config
  that should be passed to ``tools/train.py``.
* ``target_overall`` / ``target_peak`` are percentages. Use ``None`` when
  the dataset is CSRR-only (no AMR-Benchmark reference number); the
  orchestrator will then mark the run as ``measured`` instead of
  ``pass``/``fail``.
* ``target_best_snr`` is in dB.
* ``notes`` are free-form string flags that propagate to the tracking
  table (e.g. ``"AP input (CSRR variant)"``).
"""

from __future__ import annotations

from typing import Any


DATASETS: tuple[str, ...] = (
    "deepsig201610A",
    "deepsig201610B",
    "deepsig201801A",
    "hisar2019",
)


def _entry(
    config: str,
    target_overall: float | None,
    target_peak: float | None,
    target_best_snr: float | None,
    notes: str = "",
) -> dict[str, Any]:
    return dict(
        config=config,
        target_overall=target_overall,
        target_peak=target_peak,
        target_best_snr=target_best_snr,
        notes=notes,
    )


MATRIX: dict[str, dict[str, dict[str, Any]]] = {
    "cnn2": {
        "deepsig201610A": _entry("configs/cnn2/cnn2_iq-deepsig-201610A.py", 59.0, 79.0, 6),
        "deepsig201610B": _entry("configs/cnn2/cnn2_iq-deepsig-201610B.py", 64.0, 85.0, 4),
        "deepsig201801A": _entry("configs/cnn2/cnn2_iq-deepsig-201801A.py", 58.0, 92.0, 18),
        "hisar2019":      _entry("configs/cnn2/cnn2_iq-hisar-2019.py",       75.0, 100.0, 10),
    },
    "cnn4": {
        "deepsig201610A": _entry("configs/cnn4/cnn4_iq-deepsig-201610A.py", 58.0, 80.0, 4),
        "deepsig201610B": _entry("configs/cnn4/cnn4_iq-deepsig-201610B.py", 63.0, 84.0, 2),
        "deepsig201801A": _entry("configs/cnn4/cnn4_iq-deepsig-201801A.py", 55.0, 91.0, 18),
        "hisar2019":      _entry("configs/cnn4/cnn4_iq-hisar-2019.py",       70.0, 98.0, 10),
    },
    "mcnet": {
        "deepsig201610A": _entry("configs/mcnet/mcnet_iq-deepsig-201610A.py", 58.0, 82.0, 6),
        "deepsig201610B": _entry("configs/mcnet/mcnet_iq-deepsig-201610B.py", 62.0, 87.0, 4),
        "deepsig201801A": _entry("configs/mcnet/mcnet_iq-deepsig-201801A.py", 55.0, 92.0, 18),
        "hisar2019":      _entry("configs/mcnet/mcnet_iq-hisar-2019.py",       70.0, 97.0, 10),
    },
    "icamcnet": {
        "deepsig201610A": _entry("configs/icamcnet/icamcnet_iq-deepsig-201610A.py", 57.0, 83.0, 6),
        "deepsig201610B": _entry("configs/icamcnet/icamcnet_iq-deepsig-201610B.py", 62.0, 87.0, 4),
        "deepsig201801A": _entry("configs/icamcnet/icamcnet_iq-deepsig-201801A.py", 58.0, 92.0, 18),
        "hisar2019":      _entry("configs/icamcnet/icamcnet_iq-hisar-2019.py",       80.0, 100.0, 10),
    },
    "resnetamr": {
        "deepsig201610A": _entry("configs/resnetamr/resnetamr_iq-deepsig-201610A.py", 57.0, 83.0, 6),
        "deepsig201610B": _entry("configs/resnetamr/resnetamr_iq-deepsig-201610B.py", 62.0, 87.0, 4),
        "deepsig201801A": _entry("configs/resnetamr/resnetamr_iq-deepsig-201801A.py", 57.0, 91.0, 18),
        "hisar2019":      _entry("configs/resnetamr/resnetamr_iq-hisar-2019.py",       80.0, 100.0, 10),
    },
    "denscnn": {
        "deepsig201610A": _entry("configs/denscnn/denscnn_iq-deepsig-201610A.py", 57.0, 83.0, 6),
        "deepsig201610B": _entry("configs/denscnn/denscnn_iq-deepsig-201610B.py", 62.0, 87.0, 4),
        "deepsig201801A": _entry("configs/denscnn/denscnn_iq-deepsig-201801A.py", 58.0, 92.0, 18),
        "hisar2019":      _entry("configs/denscnn/denscnn_iq-hisar-2019.py",       80.0, 100.0, 10),
    },
    "gru2": {
        "deepsig201610A": _entry("configs/gru2/gru2_iq-shape-L-F-deepsig-201610A.py", 58.0, 85.0, 4),
        "deepsig201610B": _entry("configs/gru2/gru2_iq-shape-L-F-deepsig-201610B.py", 63.0, 91.0, 2),
        "deepsig201801A": _entry("configs/gru2/gru2_iq-shape-L-F-deepsig-201801A.py", 59.0, 95.0, 18),
        "hisar2019":      _entry("configs/gru2/gru2_iq-shape-L-F-hisar-2019.py",       73.0, 98.0, 10),
    },
    "lstm2": {
        "deepsig201610A": _entry("configs/lstm2/lstm2_ap-shape-L-F-deepsig-201610A.py", 58.0, 87.0, 4,
                                  notes="CSRR uses A/P (AMR-Benchmark Keras uses I/Q)"),
        "deepsig201610B": _entry("configs/lstm2/lstm2_ap-shape-L-F-deepsig-201610B.py", 64.0, 94.0, 18,
                                  notes="CSRR uses A/P (AMR-Benchmark Keras uses I/Q)"),
        "deepsig201801A": _entry("configs/lstm2/lstm2_ap-shape-L-F-deepsig-201801A.py", 60.0, 98.0, 22,
                                  notes="CSRR uses A/P (AMR-Benchmark Keras uses I/Q)"),
        "hisar2019":      _entry("configs/lstm2/lstm2_ap-shape-L-F-hisar-2019.py",       73.0, 98.0, 10,
                                  notes="CSRR uses A/P (AMR-Benchmark Keras uses I/Q)"),
    },
    "dae": {
        "deepsig201610A": _entry("configs/dae/dae_ap-deepsig-201610A.py", 57.0, 82.0, 6),
        "deepsig201610B": _entry("configs/dae/dae_ap-deepsig-201610B.py", 62.0, 85.0, 4),
        "deepsig201801A": _entry("configs/dae/dae_ap-deepsig-201801A.py", 55.0, 90.0, 18),
        "hisar2019":      _entry("configs/dae/dae_ap-hisar-2019.py",       40.0, 70.0, 10),
    },
    "mcldnn": {
        "deepsig201610A": _entry("configs/mcldnn/mcldnn_iq-deepsig-201610A.py", 62.0, 92.05, 10),
        "deepsig201610B": _entry("configs/mcldnn/mcldnn_iq-deepsig-201610B.py", 65.0, 93.0, 4),
        "deepsig201801A": _entry("configs/mcldnn/mcldnn_iq-deepsig-201801A.py", 60.0, 95.0, 18),
        "hisar2019":      _entry("configs/mcldnn/mcldnn_iq-hisar-2019.py",       75.0, 99.0, 10),
    },
    "cldnnw": {
        "deepsig201610A": _entry("configs/cldnnw/cldnnw_iq-deepsig-201610A.py", 57.0, 85.0, 6,
                                  notes="CSRR drops ZeroPadding2D (intentional)"),
        "deepsig201610B": _entry("configs/cldnnw/cldnnw_iq-deepsig-201610B.py", 62.0, 89.0, 4,
                                  notes="CSRR drops ZeroPadding2D (intentional)"),
        "deepsig201801A": _entry("configs/cldnnw/cldnnw_iq-deepsig-201801A.py", 55.0, 88.0, 18,
                                  notes="CSRR drops ZeroPadding2D (intentional)"),
        "hisar2019":      _entry("configs/cldnnw/cldnnw_iq-hisar-2019.py",       75.0, 98.0, 10,
                                  notes="CSRR drops ZeroPadding2D (intentional)"),
    },
    "cldnnl": {
        "deepsig201610A": _entry("configs/cldnnl/cldnnl_iq-deepsig-201610A.py", 57.0, 85.0, 4),
        "deepsig201610B": _entry("configs/cldnnl/cldnnl_iq-deepsig-201610B.py", 62.0, 89.0, 2),
        "deepsig201801A": _entry("configs/cldnnl/cldnnl_iq-deepsig-201801A.py", 57.0, 92.0, 18),
        "hisar2019":      _entry("configs/cldnnl/cldnnl_iq-hisar-2019.py",       75.0, 98.0, 10),
    },
    "cgdnet": {
        "deepsig201610A": _entry("configs/cgdnet/cgdnet_iq-deepsig-201610A.py", 58.0, 83.0, 6),
        "deepsig201610B": _entry("configs/cgdnet/cgdnet_iq-deepsig-201610B.py", 62.0, 88.0, 4),
        "deepsig201801A": _entry("configs/cgdnet/cgdnet_iq-deepsig-201801A.py", 57.0, 92.0, 18),
        "hisar2019":      _entry("configs/cgdnet/cgdnet_iq-hisar-2019.py",       None, None, 10,
                                  notes="No AMR-Benchmark reference for HisarMod"),
    },
    "petcgdnn": {
        "deepsig201610A": _entry("configs/petcgdnn/petcgdnn_iq-shape-L-F-deepsig-201610A.py", 60.0, 89.0, 6),
        "deepsig201610B": _entry("configs/petcgdnn/petcgdnn_iq-shape-L-F-deepsig-201610B.py", 63.0, 92.0, 4),
        "deepsig201801A": _entry("configs/petcgdnn/petcgdnn_iq-shape-L-F-deepsig-201801A.py", 60.0, 95.0, 18),
        "hisar2019":      _entry("configs/petcgdnn/petcgdnn_iq-shape-L-F-hisar-2019.py",       75.0, 99.0, 10),
    },
    "cnn1dpf": {
        "deepsig201610A": _entry("configs/cnn1dpf/cnn1dpf_iq-deepsig-201610A.py", 57.0, 85.0, 6,
                                  notes="CSRR feeds AP branches (paper uses raw IQ split)"),
        "deepsig201610B": _entry("configs/cnn1dpf/cnn1dpf_iq-deepsig-201610B.py", 62.0, 88.0, 4,
                                  notes="CSRR feeds AP branches (paper uses raw IQ split)"),
        "deepsig201801A": _entry("configs/cnn1dpf/cnn1dpf_iq-deepsig-201801A.py", 57.0, 91.0, 18,
                                  notes="CSRR feeds AP branches (paper uses raw IQ split)"),
        "hisar2019":      _entry("configs/cnn1dpf/cnn1dpf_iq-hisar-2019.py",       None, None, 10,
                                  notes="No AMR-Benchmark reference for HisarMod"),
    },
    # ----- Project-own methods (MLDNN / HCGDNN / FastMLDNN) -----
    # Targets are the original-paper numbers (see accuracy_targets.md "Project-own
    # methods" section), reported only on RML2016.10A; other datasets are
    # measured-only (target None). One-sided pass rule: measured >= target - tol.
    "mldnn": {
        "deepsig201610A": _entry("configs/mldnn/mldnn_iq-ap-deepsig201610A.py", 62.0, 92.0, 16,
                                  notes="IoT-J 2021; fig-read overall, approx; 50/10/40 vs paper protocol"),
        "deepsig201610B": _entry("configs/mldnn/mldnn_iq-ap-deepsig201610B.py", None, None, None,
                                  notes="measured-only (no extracted paper number)"),
        "deepsig201801A": _entry("configs/mldnn/mldnn_iq-ap-deepsig201801A.py", None, None, None,
                                  notes="measured-only (not reported by paper)"),
        "hisar2019":      _entry("configs/mldnn/mldnn_iq-ap-hisar2019.py",      None, None, None,
                                  notes="measured-only (not reported by paper)"),
    },
    "hcgdnn": {
        "deepsig201610A": _entry("configs/hcgdnn/hcgdnn_iq-deepsig-201610A.py", 64.9, 93.0, 16,
                                  notes="TWC 2022; overall from AMSCN comparison (0.649)"),
        "deepsig201610B": _entry("configs/hcgdnn/hcgdnn_iq-deepsig-201610B.py", None, None, None,
                                  notes="measured-only; fused (HCGDNNWeightsAccuracy) test"),
        "deepsig201801A": _entry("configs/hcgdnn/hcgdnn_iq-deepsig-201801A.py", None, None, None,
                                  notes="measured-only; fused test"),
        "hisar2019":      _entry("configs/hcgdnn/hcgdnn_iq-hisar-2019.py",      None, None, None,
                                  notes="measured-only; fused test"),
    },
    "fastmldnn": {
        "deepsig201610A": _entry("configs/fastmldnn/fastmldnn_iq-ap-deepsig-201610A.py", 63.24, 92.0, 16,
                                  notes="TCCN 2023; ULNN repro 63.01; FastMLDNNHead beta=0"),
        "deepsig201610B": _entry("configs/fastmldnn/fastmldnn_iq-ap-deepsig-201610B.py", None, None, None,
                                  notes="measured-only"),
        "deepsig201801A": _entry("configs/fastmldnn/fastmldnn_iq-ap-deepsig-201801A.py", None, None, None,
                                  notes="measured-only"),
        "hisar2019":      _entry("configs/fastmldnn/fastmldnn_iq-ap-hisar-2019.py",      None, None, None,
                                  notes="measured-only"),
    },
}


TOLERANCES: dict[str, float] = {
    "overall": 1.5,   # percentage points
    "peak": 1.0,      # percentage points
    "best_snr": 2.0,  # dB
}


def iter_jobs(models: list[str] | None = None,
              datasets: list[str] | None = None):
    """Yield ``(model, dataset, entry)`` tuples filtered by optional lists.

    Filters are case-insensitive. ``None``/empty means "all".
    """
    wanted_models = {m.lower() for m in models} if models else None
    wanted_datasets = {d.lower() for d in datasets} if datasets else None

    for model, per_dataset in MATRIX.items():
        if wanted_models is not None and model.lower() not in wanted_models:
            continue
        for dataset, entry in per_dataset.items():
            if (wanted_datasets is not None and
                    dataset.lower() not in wanted_datasets):
                continue
            yield model, dataset, entry


def known_models() -> tuple[str, ...]:
    return tuple(MATRIX.keys())


def known_datasets() -> tuple[str, ...]:
    return DATASETS
