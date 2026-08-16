#!/usr/bin/env python3
"""Family C: architecture-invariance check (inference-only, CPU).

Per run: A_hard(b) = acc(sm(tZ)[tR==b]); nP = per_bin("aff", vZ,vY,vR, tZ,tR)
(authoritative implementation: ../ladder/ladder_lib.py, fit_aff mi=60);
A_ro(b) = acc(nP[tR==b]). Emits per-bin CSV rows + per-bin test-bootstrap SE (B=200).
Val per-bin sampling cap 50k (only applied if a bin exceeds it; flagged in CSV).
Usage: python arch_invariance.py {group1|group2}
(the two run-directory groups below list the exact runs audited for the paper;
adjust the work_dirs-relative paths to where your training runs live)
"""
import sys, os, time
import numpy as np
import multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ladder"))
import ladder_lib as CA
CA.FT["aff"] = lambda Z, y: CA.fit_aff(Z, y, mi=60)

GROUP = sys.argv[1] if len(sys.argv) > 1 else ""
if GROUP not in ("group1", "group2"):
    sys.exit(__doc__)
W = "work_dirs"
OUT = f"work_dirs/arch_invariance_curves_{GROUP}.csv"
VAL_CAP = 50000
NBOOT = 200

def R(ds, bb, tier, grp, rel, seeds, cost):
    return [(ds, bb, tier, grp, s, f"{W}/{rel}/seed_{s}", cost) for s in seeds]

GROUP1 = (
    R("deepsig201801A", "mcformer", "A", "eb_2018A", "eb_2018A/amc/deepsig201801A/mcformer_hard-ce", [2026, 2027, 2028], 100)
    + R("deepsig201801A", "petcgdnn", "A", "eb_2018A", "eb_2018A/amc/deepsig201801A/petcgdnn_hard-ce", [2026, 2027, 2028], 100)
    + R("deepsig201610A", "mcformer", "A", "baseline_gate_v2", "baseline_gate_v2/amc/deepsig201610A/mcformer_hard-ce", [2026, 2027, 2028], 10)
    + R("deepsig201610A", "cgdnet", "A", "baseline_gate_v2", "baseline_gate_v2/amc/deepsig201610A/cgdnet_hard-ce", [2028], 10)  # 2026/2027 in group2 (2027 is an md5-identical mirror)
    + R("ucsd_rml22", "cnn4", "A", "ec2_rml22", "ec2_rml22/amc/ucsd_rml22/cnn4_hard-ce", [2026, 2027, 2028], 10)
    + R("ucsd_rml22", "mcformer", "A", "ec2_rml22", "ec2_rml22/amc/ucsd_rml22/mcformer_hard-ce", [2026, 2027, 2028], 10)
    + R("ucsd_rml22", "petcgdnn", "A", "ec2_rml22", "ec2_rml22/amc/ucsd_rml22/petcgdnn_hard-ce", [2026, 2027, 2028], 10)
    + R("deepsig201610B", "mcformer", "A", "ec_mcformer_10B", "ec_mcformer_10B/amc/deepsig201610B/mcformer_hard-ce", [2026, 2027, 2028], 10)
    + R("deepsig201610B", "petcgdnn", "A", "secondsurv_100ep", "secondsurv_petcgdnn10b_hard_100ep/amc/deepsig201610B/petcgdnn_hard-ce", [2026, 2027], 10)
)

GROUP2 = (
    R("hisar2019", "dscldnn", "A", "targeted_gate_20ep", "targeted_hisar_dscldnn_gate_20ep/amc/hisar2019/dscldnn_hard-ce", [2026, 2027, 2028], 50)
    + R("hisar2019", "denscnn", "A", "next_stage_20ep", "baseline_gate_next_stage_hisar_denscnn_20ep/amc/hisar2019/denscnn_hisar_hard-ce", [2026, 2027, 2028], 50)
    + R("hisar2019", "petcgdnn", "A", "next_stage_20ep_cache", "baseline_gate_next_stage_hisar_petcgdnn_20ep_cache/amc/hisar2019/petcgdnn_hisar_hard-ce", [2026, 2027, 2028], 50)
    + R("hisar2019", "mcldnn", "A", "pilot_20ep_fresh", "pilot_hisar_mcldnn_corrected_20ep_fresh/amc/hisar2019/mcldnn_hisar_hard-ce", [2026, 2027, 2028], 50)
    + R("deepsig201610A", "mldnn", "A", "baseline_gate", "baseline_gate/amc/deepsig201610A/mldnn_hard-ce", [2026, 2027, 2028], 10)
    + R("deepsig201610A", "petcgdnn", "A", "baseline_gate_v2", "baseline_gate_v2/amc/deepsig201610A/petcgdnn_hard-ce", [2026, 2027, 2028], 10)
    + R("deepsig201610A", "cgdnet", "A", "baseline_gate_v2", "baseline_gate_v2/amc/deepsig201610A/cgdnet_hard-ce", [2026, 2027], 10)
    + R("deepsig201610A", "cnn4", "B", "classic_screen", "baseline_gate_classic_screen/amc/deepsig201610A/cnn4_hard-ce", [2026], 10)
    + R("deepsig201610A", "gru2", "B", "classic_screen", "baseline_gate_classic_screen/amc/deepsig201610A/gru2_hard-ce", [2026], 10)
    + R("deepsig201610A", "cldnnw", "B", "classic_screen", "baseline_gate_classic_screen/amc/deepsig201610A/cldnnw_hard-ce", [2026], 10)
    + R("deepsig201610A", "lstm2", "B", "classic_screen2", "baseline_gate_classic_screen2/amc/deepsig201610A/lstm2_hard-ce", [2026], 10)
    + R("deepsig201610A", "cgdnet", "B", "main_10ep_3seed", "main_10ep_3seed/amc/deepsig201610A/cgdnet_hard-ce_hard", [2026, 2027, 2028], 10)
    + R("deepsig201610A", "mcldnn", "B", "main_10ep_3seed", "main_10ep_3seed/amc/deepsig201610A/mcldnn_hard-ce_hard", [2026, 2027, 2028], 10)
    + R("deepsig201610B", "petcgdnn", "A", "next_stage_10B_20ep_cache", "baseline_gate_next_stage_10B_petcgdnn_20ep_cache/amc/deepsig201610B/petcgdnn_10b_hard-ce", [2026, 2027, 2028], 10)
)

TASKS = sorted(GROUP1 if GROUP == "group1" else GROUP2, key=lambda t: -t[6])


def one(task):
    ds, bb, tier, grp, seed, rd, _ = task
    t0 = time.time()
    try:
        vZ, vY, vR = CA.load(rd + "/predictions/validation.pkl")
        tZ, tY, tR = CA.load(rd + "/predictions/test.pkl")
    except Exception as e:
        return ("ERR", f"{ds}/{bb}/{grp}/seed_{seed}: {e!r}", [])
    rng = np.random.default_rng(20260815 + seed)
    capped = 0
    keep = []
    for b in np.unique(vR):
        ii = np.where(vR == b)[0]
        if ii.size > VAL_CAP:
            ii = rng.choice(ii, VAL_CAP, replace=False)
            capped = 1
        keep.append(ii)
    if capped:
        ii = np.concatenate(keep)
        vZ, vY, vR = vZ[ii], vY[ii], vR[ii]
    hp = CA.sm(tZ)
    nP = CA.per_bin("aff", vZ, vY, vR, tZ, tR)
    hc = (hp.argmax(1) == tY).astype(float)
    rc = (nP.argmax(1) == tY).astype(float)
    rows = []
    for b in np.unique(tR):
        m = tR == b
        n = int(m.sum())
        hcm, rcm = hc[m], rc[m]
        bs = rng.integers(0, n, size=(NBOOT, n))
        seh = float(100 * hcm[bs].mean(1).std(ddof=1))
        ser = float(100 * rcm[bs].mean(1).std(ddof=1))
        rows.append(f"{ds},{bb},{tier},{seed},{b:g},{100*hcm.mean():.4f},{100*rcm.mean():.4f},{n},{grp},{GROUP},{seh:.4f},{ser:.4f},{capped}")
    msg = f"{ds}/{bb}/{grp}/seed_{seed}: OK bins={len(rows)} allsnr_hard={100*hc.mean():.2f} allsnr_ro={100*rc.mean():.2f} capped={capped} {time.time()-t0:.0f}s"
    return ("OK", msg, rows)


def main():
    np_proc = min(12, os.cpu_count() or 4)
    print(f"[{GROUP}] {len(TASKS)} runs, pool={np_proc}", flush=True)
    t0 = time.time()
    with open(OUT, "w") as f:
        f.write("dataset,backbone,tier,seed,bin,acc_hard,acc_readout,n,group,server,se_hard,se_ro,val_capped\n")
        with mp.Pool(np_proc) as pool:
            for st, msg, rows in pool.imap_unordered(one, TASKS):
                print(f"[{st}] {msg}", flush=True)
                for r in rows:
                    f.write(r + "\n")
                f.flush()
    print(f"[{GROUP}] DONE {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
