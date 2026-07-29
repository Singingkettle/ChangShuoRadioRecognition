auto_seed_next_near_miss_wave() {
    # Cap/rate-limit auto-seed (stall class auto_seed_wave_spam_phantom):
    # - at most ONE new wave ahead of the highest non-junk wave
    # - refuse while pending OR stale running remains
    # - require distinct wave-suffixed variants that lack retune_results rows
    # - never reopen exhausted auto-seed entries; never phantom-exhaust empty waves
    "${PY}" - "${QUEUE_FULL}" "${REPO}" <<'PY'
import json, re, sys
from datetime import datetime, timezone
from pathlib import Path

queue_path = Path(sys.argv[1])
repo = Path(sys.argv[2])
cfg = repo / "configs/amr_benchmark/retune"
stamp = repo / "work_dirs/amr_benchmark_retune/auto_seed_stamp.json"
results_md = repo / "docs/amr_benchmark/retune_results.md"
data = json.loads(queue_path.read_text())
now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
MAX_AHEAD = 1
MIN_SEED_INTERVAL_SEC = 6 * 3600  # rate-limit: at most one auto-seed per 6h

def wave_num(eid: str):
    m = re.search(r"wave(\d+)", eid or "")
    return int(m.group(1)) if m else None

entries = data.get("entries", [])
active = [e for e in entries if e.get("status", "pending") in {"pending", "running"}]
if active:
    ids = ",".join(e.get("id", "?") for e in active[:8])
    print(f"auto_seed_blocked_active:{ids}")
    sys.exit(1)

# Highest "real" wave = hand-seeded / non-auto-junk.
real_waves = []
junk_waves = []
for e in entries:
    n = wave_num(e.get("id", ""))
    if n is None:
        continue
    notes = e.get("notes") or ""
    is_auto_junk = ("auto_seed_next_near_miss" in notes) and e.get("status") == "exhausted"
    if is_auto_junk:
        junk_waves.append(n)
    else:
        real_waves.append(n)

anchor = max(real_waves) if real_waves else 9
nxt = anchor + MAX_AHEAD
# If a non-junk entry already exists at/above nxt, do not advance further.
if any(n >= nxt for n in real_waves if n > anchor):
    print(f"auto_seed_capped_anchor{anchor}_already_ahead")
    sys.exit(3)
# Ignore junk wave numbers (wave10–81 spam) when choosing nxt; never chase max(junk).

# Rate-limit stamp
if stamp.is_file():
    try:
        prev = json.loads(stamp.read_text())
        prev_t = datetime.fromisoformat(
            prev.get("at", "1970-01-01T00:00:00+00:00").replace("Z", "+00:00")
        )
        age = (datetime.now(timezone.utc) - prev_t).total_seconds()
        if age < MIN_SEED_INTERVAL_SEC and int(prev.get("wave", -1)) == nxt:
            print(f"auto_seed_rate_limited_age{int(age)}s_wave{nxt}")
            sys.exit(4)
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        pass

results_text = results_md.read_text(errors="replace") if results_md.is_file() else ""

def variant_has_result(model: str, dataset: str, variant: str) -> bool:
    label = f"{model}/{dataset}"
    return bool(re.search(
        r"\|\s*[0-9-]+\s+[0-9:]+\s*\|\s*"
        + re.escape(label)
        + r"\s*\|\s*`"
        + re.escape(variant)
        + r"`\s*\|",
        results_text,
        re.I,
    ))

def write_new(path: Path, body: str) -> None:
    path.write_text(body)

def generate_wave_artifacts(n: int):
    """Create DISTINCT wave-suffixed Tier-A FT recipes (never reuse bare variant names)."""
    fast_ckpt = repo / (
        "work_dirs/amr_benchmark_retune/fastmldnn/deepsig201610A/"
        "paper_fixedlr_l2_ft80_from_w8best/best_accuracy_top1_epoch_37.pth"
    )
    if not fast_ckpt.is_file():
        cands = sorted(
            (repo / "work_dirs/amr_benchmark_retune/fastmldnn/deepsig201610A").glob(
                "*/best_accuracy_top1_epoch_*.pth"
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        fast_ckpt = cands[0] if cands else None
    hcg_ckpt = repo / (
        "work_dirs/amr_benchmark_retune/hcgdnn/deepsig201610A/"
        "paper_multistep_exact800_esoff1600/best_accuracy_top1_epoch_968.pth"
    )
    if not hcg_ckpt.is_file():
        cands = sorted(
            (repo / "work_dirs/amr_benchmark_retune/hcgdnn/deepsig201610A").glob(
                "*/best_accuracy_top1_epoch_*.pth"
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        hcg_ckpt = cands[0] if cands else None

    fast_specs = []
    if fast_ckpt and fast_ckpt.is_file():
        rel = fast_ckpt.relative_to(repo).as_posix()
        v = f"paper_fixedlr_l2_ft120_lr5e5_from_w9best_w{n}"
        write_new(cfg / f"wave{n}_fastmldnn_deepsig201610A_{v}.py", f"""\"\"\"Wave-{n} auto: FastMLDNN gentle FT 120ep @ lr5e-5 from W9 best (arch freeze).\"\"\"
_base_ = ['./wave6_fastmldnn_deepsig201610A_paper_fixedlr_l2_esoff500.py']
load_from = '{rel}'
optim_wrapper = dict(optimizer=dict(type='Adam', lr=5e-5, weight_decay=1e-4))
train_cfg = dict(by_epoch=True, max_epochs=120, val_interval=1)
param_scheduler = dict(_delete_=True, type='CosineAnnealingLR', by_epoch=True, T_max=120, eta_min=1e-6)
""")
        fast_specs.append((v, 0, f"gentle FT from W9 best ckpt wave{n}"))

    hcg_specs = []
    if hcg_ckpt and hcg_ckpt.is_file():
        rel = hcg_ckpt.relative_to(repo).as_posix()
        v = f"paper_multistep_l2_ft120_lr5e5_from_w9best_w{n}"
        write_new(cfg / f"wave{n}_hcgdnn_deepsig201610A_{v}.py", f"""\"\"\"Wave-{n} auto: HCGDNN gentle FT 120ep @ lr5e-5 from W9 best 63.30 (arch freeze).\"\"\"
_base_ = ['./wave7_hcgdnn_deepsig201610A_paper_multistep_l2_esoff400.py']
load_from = '{rel}'
optim_wrapper = dict(optimizer=dict(type='Adam', lr=5e-5, weight_decay=1e-4))
train_cfg = dict(by_epoch=True, max_epochs=120, val_interval=1)
param_scheduler = dict(_delete_=True, type='CosineAnnealingLR', by_epoch=True, T_max=120, eta_min=1e-6)
""")
        hcg_specs.append((v, 0, f"gentle FT from W9 HCG 63.30 wave{n}"))

    fast_specs = [s for s in fast_specs if not variant_has_result("fastmldnn", "deepsig201610A", s[0])]
    hcg_specs = [s for s in hcg_specs if not variant_has_result("hcgdnn", "deepsig201610A", s[0])]
    if not fast_specs and not hcg_specs:
        return [], []

    def man_exps(model, specs):
        exps = []
        for i, (variant, pri, notes) in enumerate(specs):
            conf = f"configs/amr_benchmark/retune/wave{n}_{model}_deepsig201610A_{variant}.py"
            exps.append({
                "id": f"W{n}_{model}_10A_{variant}"[:80],
                "priority": pri if pri is not None else i,
                "model": model, "dataset": "deepsig201610A", "variant": variant,
                "config": conf, "notes": notes,
            })
        return exps

    if fast_specs:
        (cfg / f"siege_fastmldnn_10a_wave{n}.json").write_text(json.dumps({
            "description": f"Wave-{n} Tier-A FastMLDNN auto (capped; distinct FT only)",
            "updated_at": now, "campaign_mode": "paper_exact",
            "paper_target_overall": 63.24, "paper_target_peak": 92.0,
            "experiments": man_exps("fastmldnn", fast_specs),
        }, indent=2) + "\n")
    if hcg_specs:
        (cfg / f"siege_hcgdnn_10a_wave{n}.json").write_text(json.dumps({
            "description": f"Wave-{n} Tier-A HCGDNN auto (capped; distinct FT only)",
            "updated_at": now, "campaign_mode": "paper_exact",
            "paper_target_overall": 64.9, "paper_target_peak": 93.0,
            "experiments": man_exps("hcgdnn", hcg_specs),
        }, indent=2) + "\n")
    return fast_specs, hcg_specs

fast_man = cfg / f"siege_fastmldnn_10a_wave{nxt}.json"
hcg_man = cfg / f"siege_hcgdnn_10a_wave{nxt}.json"
need_gen = True
if fast_man.is_file() and hcg_man.is_file():
    try:
        fe = json.loads(fast_man.read_text()).get("experiments") or []
        he = json.loads(hcg_man.read_text()).get("experiments") or []

        def useful(exps, model):
            for e in exps:
                v = e.get("variant") or ""
                if v.endswith(f"_w{nxt}") and not variant_has_result(model, "deepsig201610A", v):
                    return True
            return False

        if useful(fe, "fastmldnn") or useful(he, "hcgdnn"):
            need_gen = False
    except (OSError, json.JSONDecodeError):
        need_gen = True

if need_gen:
    for p in (fast_man, hcg_man, cfg / f"siege_tierb_wave{nxt}.json"):
        if p.is_file():
            p.unlink()
    fast_specs, hcg_specs = generate_wave_artifacts(nxt)
    if not fast_specs and not hcg_specs:
        print("auto_seed_no_distinct_configs")
        sys.exit(5)

candidates = []
if (cfg / f"siege_fastmldnn_10a_wave{nxt}.json").is_file():
    candidates.append((
        f"siege_fastmldnn_10a_wave{nxt}", f"siege_fastmldnn_10a_wave{nxt}.json",
        "fastmldnn", "deepsig201610A", 0, 0.35,
    ))
if (cfg / f"siege_hcgdnn_10a_wave{nxt}.json").is_file():
    candidates.append((
        f"siege_hcgdnn_10a_wave{nxt}", f"siege_hcgdnn_10a_wave{nxt}.json",
        "hcgdnn", "deepsig201610A", 0, 1.6,
    ))

added = []
ids = {e.get("id") for e in entries}
for eid, fname, model, dataset, pri, gap in candidates:
    man = cfg / fname
    if not man.is_file() or eid in ids:
        continue
    try:
        man_data = json.loads(man.read_text())
    except (OSError, json.JSONDecodeError):
        continue
    exps = man_data.get("experiments") or []
    if not any(not variant_has_result(model, dataset, e.get("variant") or "") for e in exps):
        continue
    data.setdefault("entries", []).append({
        "id": eid,
        "priority": pri,
        "model": model,
        "dataset": dataset,
        "manifest": f"configs/amr_benchmark/retune/{fname}",
        "status": "pending",
        "gap_pp": gap,
        "failing_metric": "overall",
        "notes": f"auto_seed_capped_wave{nxt} {now} max_ahead={MAX_AHEAD}",
    })
    added.append(eid)

if not added:
    print("auto_seed_no_new_pending")
    sys.exit(6)

data["updated_at"] = now
note = data.get("notes") or ""
data["notes"] = note + f" | auto_seed_capped_wave{nxt}_{now} " + ",".join(added)
queue_path.write_text(json.dumps(data, indent=2) + "\n")
stamp.parent.mkdir(parents=True, exist_ok=True)
stamp.write_text(json.dumps({"at": now, "wave": nxt, "added": added}, indent=2) + "\n")
print(",".join(added))
sys.exit(0)
PY
}
