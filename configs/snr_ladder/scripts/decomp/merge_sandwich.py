#!/usr/bin/env python3
"""Merge per-dataset E1/E2 CSVs and CH CSVs into bayes_sandwich.csv (+extended)."""
import csv, json, os, sys

SRC = sys.argv[1]           # dir with *_e1e2.csv, *_ch.csv, *_meta.json
OUT = sys.argv[2]           # ceiling_campaign dir
DATASETS = ["deepsig201610A", "ucsd_rml22", "deepsig201610B", "deepsig201801A"]

def read_csv(p):
    if not os.path.exists(p): return []
    with open(p) as f: return list(csv.DictReader(f))

def tag_from_upper(up):
    if up is None: return "no_phi"
    if up >= 90.0: return "trusted"
    if up >= 70.0: return "biased+3to4"
    return "biased+7to8"

rows_main, rows_ext = [], []
for ds in DATASETS:
    e = read_csv(os.path.join(SRC, f"{ds}_e1e2.csv"))
    ch = {float(r["bin"]): r for r in read_csv(os.path.join(SRC, f"{ds}_ch.csv"))}
    meta = json.load(open(os.path.join(SRC, f"{ds}_meta.json")))
    for r in e:
        b = float(r["bin"])
        c = ch.get(b)
        up = float(c["CH_upper"]) if c else None
        lo = float(c["CH_lower_1mR15"]) if c else None
        tag = tag_from_upper(up)
        rows_main.append([ds, f"{b:g}", r["n_test"], r["E1_best_single_readout"],
                          r["E2_stacked"],
                          f"{lo:.3f}" if lo is not None else "",
                          f"{up:.3f}" if up is not None else "", tag])
        rows_ext.append([ds, f"{b:g}", r["n_test"], r["E1_best_single_readout"], r["e1_arch"],
                         r["E2_stacked"], r["in_band"],
                         c["R1"] if c else "", c["R15"] if c else "",
                         c["n_bin"] if c else "", c["phi"] if c else "", meta["best_arch"],
                         "|".join(meta["archs_kept"])])

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "bayes_sandwich.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "bin", "n_test", "E1_best_single_readout", "E2_stacked",
                "CH_lower_1mR15", "CH_upper", "ch_reliability_tag"])
    w.writerows(rows_main)
with open(os.path.join(OUT, "bayes_sandwich_extended.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "bin", "n_test", "E1_best_single_readout", "e1_arch", "E2_stacked",
                "in_band", "R1", "R15", "n_phi_bin", "phi_file", "best_arch_overall",
                "archs_stacked"])
    w.writerows(rows_ext)

# band summaries
print("dataset | band_bins | n-weighted band E1 | E2 | E2-E1 | CH interval (band, n-weighted)")
for ds in DATASETS:
    e = read_csv(os.path.join(SRC, f"{ds}_e1e2.csv"))
    ch = {float(r["bin"]): r for r in read_csv(os.path.join(SRC, f"{ds}_ch.csv"))}
    band = [r for r in e if r["in_band"] == "1"]
    if not band: continue
    n = sum(int(r["n_test"]) for r in band)
    e1 = sum(float(r["E1_best_single_readout"]) * int(r["n_test"]) for r in band) / n
    e2 = sum(float(r["E2_stacked"]) * int(r["n_test"]) for r in band) / n if band[0]["E2_stacked"] else float("nan")
    if ch:
        lo = sum(float(ch[float(r["bin"])]["CH_lower_1mR15"]) * int(r["n_test"]) for r in band) / n
        up = sum(float(ch[float(r["bin"])]["CH_upper"]) * int(r["n_test"]) for r in band) / n
        chs = f"[{lo:.2f}, {up:.2f}]"
    else:
        chs = "no_phi"
    bins = ",".join(f'{float(r["bin"]):g}' for r in band)
    print(f"{ds} | {bins} | {e1:.2f} | {e2:.2f} | {e2-e1:+.2f} | {chs}")
