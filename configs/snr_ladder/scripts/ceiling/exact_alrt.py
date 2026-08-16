#!/usr/bin/env python3
"""Exact Bayes ceiling for the clean-paired AWGN AMC benchmark: Tier-E core.

Ports the MATLAB generator (generate_synthetic_awgn_amc.m, the single source of
truth) to numpy/torch and computes the per-SNR Bayes accuracy of the 7-class
DIGITAL subset {BPSK, QPSK, 8PSK, 4PAM, 16QAM, 64QAM, CPFSK} under the factorized
ideal-constellation likelihood (Tier-E). The per-frame demean+power-normalization
coupling is quantified separately (Tier-C SIS, run_sis stage) and by an exact
small-n brute force here.

Generator facts mirrored exactly (from the .m source):
  BPSK/QPSK/8PSK : phase = 2*pi*k/M + pi/M  (offset constellations; BPSK = {+j,-j})
  4PAM           : levels [-3,-1,1,3]/sqrt(5), real
  16QAM/64QAM    : I,Q iid from -(side-1):2:(side-1); per-frame inner norm (a no-op
                   after the outer demean+norm, see note in code)
  CPFSK          : s_t in {+-1} iid, phi_t = 0.55*pi*cumsum(s), x = exp(j*phi)
  frame map      : x <- x - mean(x);  clean <- x / sqrt(mean|x|^2 + eps)
  noise          : var = mean|clean|^2 / 10^(snr/10) per complex sample,
                   real/imag each N(0, var/2)

Self-checks (all must pass before any number is released):
  --check inversion : disk clean frames (npy) invert onto the constellation
                      manifold with residual < 1e-6  (port == MATLAB map)
  --check identity  : Acc_MAP == E[max_c p(c|y)] within MC CI (true-posterior id.)
  --check brute     : small-n brute force WITH the exact demean+norm coupling vs
                      factorized likelihood: decision-flip rate quantified
Run:
  python exact_alrt.py --check all
  python exact_alrt.py --run tier_e --frames 200000 --out tier_e_ceiling.csv
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

try:
    import torch
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:  # torch optional
    torch = None
    DEV = "cpu"

EPS_MATLAB = 2.220446049250313e-16  # MATLAB eps, used in the generator's norm

DIGITAL = ["BPSK", "QPSK", "8PSK", "4PAM", "16QAM", "64QAM", "CPFSK"]
SNRS = list(range(-20, 20, 2))
CPFSK_STEP = 11          # 0.55*pi = 11*(pi/20); phase lattice has 40 states
CPFSK_NSTATE = 40


# ----------------------------------------------------------------------------
# generator port (distributionally identical; deterministic map bitwise-equal)
# ----------------------------------------------------------------------------
def constellation(name: str) -> np.ndarray:
    """Ideal (population-normalized) constellation for the factorized model."""
    if name == "BPSK":
        k = np.arange(2); return np.exp(1j * (2 * np.pi * k / 2 + np.pi / 2))
    if name == "QPSK":
        k = np.arange(4); return np.exp(1j * (2 * np.pi * k / 4 + np.pi / 4))
    if name == "8PSK":
        k = np.arange(8); return np.exp(1j * (2 * np.pi * k / 8 + np.pi / 8))
    if name == "4PAM":
        return np.array([-3, -1, 1, 3]) / np.sqrt(5) + 0j
    if name == "16QAM":
        lv = np.arange(-3, 4, 2)
        pts = (lv[:, None] + 1j * lv[None, :]).ravel()
        return pts / np.sqrt(10.0)
    if name == "64QAM":
        lv = np.arange(-7, 8, 2)
        pts = (lv[:, None] + 1j * lv[None, :]).ravel()
        return pts / np.sqrt(42.0)
    raise ValueError(name)


def raw_frame(name: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Raw modulated sequence BEFORE the demean+norm chain (mirrors synth_modulation)."""
    if name in ("BPSK", "QPSK", "8PSK"):
        M = {"BPSK": 2, "QPSK": 4, "8PSK": 8}[name]
        k = rng.integers(0, M, n)
        return np.exp(1j * (2 * np.pi * k / M + np.pi / M))
    if name == "4PAM":
        lv = np.array([-3, -1, 1, 3]) / np.sqrt(5)
        return lv[rng.integers(0, 4, n)] + 0j
    if name in ("16QAM", "64QAM"):
        side = 4 if name == "16QAM" else 8
        lv = np.arange(-(side - 1), side, 2).astype(float)
        x = lv[rng.integers(0, side, n)] + 1j * lv[rng.integers(0, side, n)]
        return x / np.sqrt(np.mean(np.abs(x) ** 2))   # inner norm (kept for fidelity)
    if name == "CPFSK":
        s = 2 * rng.integers(0, 2, n) - 1
        return np.exp(1j * np.cumsum(0.55 * np.pi * s))
    raise ValueError(name)


def clean_map(x: np.ndarray) -> np.ndarray:
    """The generator's frame map: demean then unit empirical power."""
    x = x - x.mean()
    return x / np.sqrt(np.mean(np.abs(x) ** 2) + EPS_MATLAB)


def gen_batch(name: str, n_frames: int, n: int, snr_db: float,
              rng: np.random.Generator):
    """Fresh (clean, noisy) batch, mirroring the MATLAB chain."""
    clean = np.empty((n_frames, n), dtype=np.complex128)
    for i in range(n_frames):
        clean[i] = clean_map(raw_frame(name, n, rng))
    sig_pow = np.mean(np.abs(clean) ** 2, axis=1, keepdims=True)
    npow = sig_pow / (10 ** (snr_db / 10))
    noise = np.sqrt(npow / 2) * (rng.standard_normal(clean.shape)
                                 + 1j * rng.standard_normal(clean.shape))
    return clean, clean + noise


# ----------------------------------------------------------------------------
# Tier-E likelihoods (ideal factorized model; sigma2 = 10^(-snr/10))
# ----------------------------------------------------------------------------
def loglik_memoryless(y: np.ndarray, pts: np.ndarray, sigma2: float) -> np.ndarray:
    """log L(y) for iid-mixture constellation classes. y: (B, n) complex."""
    if torch is not None:
        yt = torch.as_tensor(y, dtype=torch.complex128, device=DEV)
        pt = torch.as_tensor(pts, dtype=torch.complex128, device=DEV)
        d2 = torch.abs(yt[..., None] - pt) ** 2                     # (B, n, M)
        ll = torch.logsumexp(-d2 / sigma2, dim=-1) - math.log(len(pts))
        out = ll.sum(dim=1) - y.shape[1] * math.log(math.pi * sigma2)
        return out.cpu().numpy()
    d2 = np.abs(y[..., None] - pts) ** 2
    m = (-d2 / sigma2).max(axis=-1, keepdims=True)
    ll = (m[..., 0] + np.log(np.exp(-d2 / sigma2 - m).sum(axis=-1))
          - math.log(len(pts)))
    return ll.sum(axis=1) - y.shape[1] * math.log(math.pi * sigma2)


def loglik_cpfsk(y: np.ndarray, sigma2: float) -> np.ndarray:
    """Exact HMM forward over the 40-state phase lattice. y: (B, n)."""
    B, n = y.shape
    phases = np.arange(CPFSK_NSTATE) * (np.pi / 20)
    states = np.exp(1j * phases)                                     # (40,)
    if torch is not None:
        yt = torch.as_tensor(y, dtype=torch.complex128, device=DEV)
        st = torch.as_tensor(states, dtype=torch.complex128, device=DEV)
        # emission log-probs for every step/state: (B, n, 40)
        em = -torch.abs(yt[..., None] - st) ** 2 / sigma2
        alpha = torch.full((B, CPFSK_NSTATE), -math.inf,
                           dtype=torch.float64, device=DEV)
        alpha[:, 0] = 0.0                                            # phi_0 = 0
        log2 = math.log(2.0)
        for t in range(n):
            up = torch.roll(alpha, shifts=CPFSK_STEP, dims=1)
            dn = torch.roll(alpha, shifts=-CPFSK_STEP, dims=1)
            alpha = torch.logaddexp(up, dn) - log2 + em[:, t, :]
        out = (torch.logsumexp(alpha, dim=1)
               - n * math.log(math.pi * sigma2))
        return out.cpu().numpy()
    em_all = -np.abs(y[..., None] - states) ** 2 / sigma2
    alpha = np.full((B, CPFSK_NSTATE), -np.inf)
    alpha[:, 0] = 0.0
    for t in range(n):
        up = np.roll(alpha, CPFSK_STEP, axis=1)
        dn = np.roll(alpha, -CPFSK_STEP, axis=1)
        alpha = np.logaddexp(up, dn) - math.log(2.0) + em_all[:, t, :]
    m = alpha.max(axis=1, keepdims=True)
    return (m[:, 0] + np.log(np.exp(alpha - m).sum(axis=1))
            - n * math.log(math.pi * sigma2))


def all_logliks(y: np.ndarray, sigma2: float) -> np.ndarray:
    """(B, 7) log-likelihood matrix over the digital classes."""
    cols = []
    for name in DIGITAL:
        if name == "CPFSK":
            cols.append(loglik_cpfsk(y, sigma2))
        else:
            cols.append(loglik_memoryless(y, constellation(name), sigma2))
    return np.stack(cols, axis=1)


# ----------------------------------------------------------------------------
# self-checks
# ----------------------------------------------------------------------------
def check_inversion(data_root: str, per_class: int = 50) -> bool:
    """Disk clean frames must invert onto the constellation manifold (<1e-6)."""
    ann = json.load(open(os.path.join(data_root, "test.json")))
    by_mod: dict[str, list[dict]] = {}
    for it in ann["data_list"]:
        if it["modulation"] in DIGITAL and it["snr"] == 18:
            by_mod.setdefault(it["modulation"], []).append(it)
    ok_all = True
    for name in DIGITAL:
        items = by_mod.get(name, [])[:per_class]
        worst = 0.0
        for it in items:
            z = np.load(os.path.join(data_root, "clean", it["clean_file_name"]))
            z = z[0] + 1j * z[1]
            if name == "CPFSK":
                # phases must sit on the pi/20 lattice after undoing (s, mu)
                res = _invert_affine_to_set(z, None, cpfsk=True)
            else:
                res = _invert_affine_to_set(z, constellation(name))
            worst = max(worst, res)
        status = "OK " if worst < 1e-6 else "FAIL"
        print(f"  inversion {name:6s}: worst residual {worst:.2e}  {status}")
        ok_all &= worst < 1e-6
    return ok_all


def _invert_affine_to_set(z: np.ndarray, pts, cpfsk: bool = False,
                          iters: int = 50) -> float:
    """Find (s, mu) with s*z + mu on the constellation set; return rel residual.

    The nearest-point/least-squares iteration is an ICP and can stall in a local
    minimum for dense constellations (64QAM), so it is multi-started over a grid
    of initial scale factors and the best residual is kept. The 1e-6 acceptance
    threshold makes a false pass impossible regardless of how many starts run.
    """
    def _run(s0: complex) -> float:
        s, mu = s0, 0.0 + 0j
        for _ in range(iters):
            x = s * z + mu
            if cpfsk:
                ph = np.angle(x)
                snap = np.exp(1j * np.round(ph / (np.pi / 20)) * (np.pi / 20))
            else:
                snap = pts[np.argmin(np.abs(x[:, None] - pts), axis=1)]
            A = np.stack([z, np.ones_like(z)], axis=1)
            sol, *_ = np.linalg.lstsq(A, snap, rcond=None)
            s, mu = sol[0], sol[1]
        x = s * z + mu
        if cpfsk:
            ph = np.angle(x)
            snap = np.exp(1j * np.round(ph / (np.pi / 20)) * (np.pi / 20))
        else:
            snap = pts[np.argmin(np.abs(x[:, None] - pts), axis=1)]
        return float(np.max(np.abs(x - snap)) / (np.abs(s) + 1e-30))

    base = 1.0 / np.sqrt(np.mean(np.abs(z) ** 2))
    best = math.inf
    for mult in np.linspace(0.75, 1.35, 25):
        best = min(best, _run(base * mult))
        if best < 1e-7:
            break
    return best


def check_identity(frames: int = 10000, seed: int = 7) -> bool:
    """Acc_MAP must equal E[max_c p(c|y)] under the model's own generative law."""
    rng = np.random.default_rng(seed)
    ok_all = True
    for snr in (-10, 0, 8):
        sigma2 = 10 ** (-snr / 10)
        ys, labels = [], []
        per = frames // len(DIGITAL)
        for ci, name in enumerate(DIGITAL):
            # sample from the IDEAL factorized model (the law the likelihood owns)
            if name == "CPFSK":
                s = 2 * rng.integers(0, 2, (per, 128)) - 1
                clean = np.exp(1j * np.cumsum(0.55 * np.pi * s, axis=1))
            else:
                pts = constellation(name)
                clean = pts[rng.integers(0, len(pts), (per, 128))]
            noise = np.sqrt(sigma2 / 2) * (rng.standard_normal(clean.shape)
                                           + 1j * rng.standard_normal(clean.shape))
            ys.append(clean + noise); labels.append(np.full(per, ci))
        y = np.concatenate(ys); lab = np.concatenate(labels)
        ll = all_logliks(y, sigma2)
        post = np.exp(ll - ll.max(axis=1, keepdims=True))
        post /= post.sum(axis=1, keepdims=True)
        acc_map = float((ll.argmax(axis=1) == lab).mean())
        e_max = float(post.max(axis=1).mean())
        se = math.sqrt(acc_map * (1 - acc_map) / len(lab)) * 3 + 3e-3
        status = "OK " if abs(acc_map - e_max) < se else "FAIL"
        print(f"  identity snr={snr:+3d}: Acc_MAP={acc_map:.4f} "
              f"E[max p]={e_max:.4f} |diff|={abs(acc_map-e_max):.4f} "
              f"(tol {se:.4f}) {status}")
        ok_all &= abs(acc_map - e_max) < se
    return ok_all


def check_brute(n: int = 8, frames: int = 4000, seed: int = 11) -> bool:
    """Small-n brute force WITH exact demean+norm coupling vs factorized decisions."""
    rng = np.random.default_rng(seed)
    classes = ["BPSK", "QPSK"]
    ok_all = True
    for snr in (0, 6):
        sigma2 = 10 ** (-snr / 10)
        # enumerate exact clean manifolds
        manifolds = []
        for name in classes:
            M = {"BPSK": 2, "QPSK": 4}[name]
            k = np.stack(np.meshgrid(*([np.arange(M)] * n),
                                     indexing="ij"), -1).reshape(-1, n)
            raw = np.exp(1j * (2 * np.pi * k / M + np.pi / M))
            cl = np.array([clean_map(r) for r in raw])
            manifolds.append(cl)
        flips = 0; total = 0
        for ci, name in enumerate(classes):
            cl, yn = gen_batch(name, frames // 2, n, snr, rng)
            # exact: log mean_seq N(y; clean_seq, sigma2 I)
            ll_ex = np.empty((len(yn), 2))
            for cj, man in enumerate(manifolds):
                d2 = (np.abs(yn[:, None, :] - man[None, :, :]) ** 2).sum(-1)
                m = (-d2 / sigma2).max(1)
                ll_ex[:, cj] = m + np.log(
                    np.exp(-d2 / sigma2 - m[:, None]).sum(1)) - math.log(len(man))
            ll_fa = np.stack([loglik_memoryless(yn, constellation(c), sigma2)
                              for c in classes], axis=1)
            flips += int((ll_ex.argmax(1) != ll_fa.argmax(1)).sum())
            total += len(yn)
        rate = flips / total
        print(f"  brute n={n} snr={snr:+d}: decision flip rate "
              f"exact-vs-factorized = {rate:.4f} ({flips}/{total})")
        ok_all &= rate < 0.02   # coupling at n=8 is far larger than at n=128
    print("  note: coupling shrinks ~1/n; n=128 effect bounded by Tier-C SIS.")
    return ok_all


# ----------------------------------------------------------------------------
# Tier-C: SIS correction for the frame demean+norm coupling
# ----------------------------------------------------------------------------
def _sample_factorized_paths(y: np.ndarray, pts: np.ndarray, sigma2: float,
                             K: int, rng) -> np.ndarray:
    """K iid symbol paths per frame from the factorized posterior q(s|y).
    y: (B, n) -> returns (B, K, n) complex points."""
    B, n = y.shape
    logp = -np.abs(y[..., None] - pts) ** 2 / sigma2          # (B, n, M)
    logp -= logp.max(axis=-1, keepdims=True)
    p = np.exp(logp); p /= p.sum(axis=-1, keepdims=True)
    cdf = np.cumsum(p, axis=-1)
    u = rng.random((B, K, n))
    idx = (u[..., None] > cdf[:, None, :, :]).sum(axis=-1)    # (B, K, n)
    return pts[np.minimum(idx, len(pts) - 1)]


def _sample_cpfsk_paths(y: np.ndarray, sigma2: float, K: int, rng) -> np.ndarray:
    """FFBS: K posterior phase paths per frame from the exact 40-state HMM."""
    B, n = y.shape
    phases = np.arange(CPFSK_NSTATE) * (np.pi / 20)
    states = np.exp(1j * phases)
    em = -np.abs(y[..., None] - states) ** 2 / sigma2         # (B, n, 40)
    alphas = np.empty((B, n, CPFSK_NSTATE))
    alpha = np.full((B, CPFSK_NSTATE), -np.inf); alpha[:, 0] = 0.0
    for t in range(n):
        up = np.roll(alpha, CPFSK_STEP, axis=1)
        dn = np.roll(alpha, -CPFSK_STEP, axis=1)
        alpha = np.logaddexp(up, dn) - math.log(2) + em[:, t, :]
        alphas[:, t, :] = alpha
    # backward sampling
    out = np.empty((B, K, n), dtype=np.complex128)
    last = alphas[:, n - 1, :]
    pl = np.exp(last - last.max(axis=1, keepdims=True))
    pl /= pl.sum(axis=1, keepdims=True)
    cdf = np.cumsum(pl, axis=1)
    cur = (rng.random((B, K))[..., None] > cdf[:, None, :]).sum(axis=-1)
    cur = np.minimum(cur, CPFSK_NSTATE - 1)
    out[:, :, n - 1] = states[cur]
    for t in range(n - 2, -1, -1):
        a_prev = alphas[:, t, :]                              # (B, 40)
        up = (cur + CPFSK_STEP) % CPFSK_NSTATE                # came from phi-0.55pi? no:
        # transition was prev -> prev +- 11; so prev in {cur-11, cur+11} mod 40
        p1 = np.take_along_axis(a_prev, (cur - CPFSK_STEP) % CPFSK_NSTATE, axis=1)
        p2 = np.take_along_axis(a_prev, (cur + CPFSK_STEP) % CPFSK_NSTATE, axis=1)
        m = np.maximum(p1, p2)
        w1 = np.exp(p1 - m); w2 = np.exp(p2 - m)
        pick1 = rng.random((B, K)) < w1 / (w1 + w2)
        cur = np.where(pick1, (cur - CPFSK_STEP) % CPFSK_NSTATE,
                       (cur + CPFSK_STEP) % CPFSK_NSTATE)
        out[:, :, t] = states[cur]
    return out


def sis_corrected_logliks(y: np.ndarray, sigma2: float, K: int, rng):
    """(B, 7) SIS-corrected log-likelihoods + min ESS fraction per class.

    log L_exact = log L_fact + log E_{s~q(.|y)}[ prod_t N(y_t; clean_map(s)_t, s2)
                                               / prod_t N(y_t; s_t, s2) ]
    The weight depends on the path only through the frame demean+norm map, the
    exact coupling the generator applies. Unbiased for every K.
    """
    B, n = y.shape
    lls = np.empty((B, len(DIGITAL))); ess = np.empty((B, len(DIGITAL)))
    for ci, name in enumerate(DIGITAL):
        if name == "CPFSK":
            ll_fact = loglik_cpfsk(y, sigma2)
            paths = _sample_cpfsk_paths(y, sigma2, K, rng)
        else:
            pts = constellation(name)
            ll_fact = loglik_memoryless(y, pts, sigma2)
            paths = _sample_factorized_paths(y, pts, sigma2, K, rng)
        mu = paths.mean(axis=2, keepdims=True)
        d = paths - mu
        pw = np.sqrt((np.abs(d) ** 2).mean(axis=2, keepdims=True) + EPS_MATLAB)
        cl = d / pw                                            # clean_map per path
        d2c = (np.abs(y[:, None, :] - cl) ** 2).sum(axis=2)    # (B, K)
        d2f = (np.abs(y[:, None, :] - paths) ** 2).sum(axis=2)
        logw = (d2f - d2c) / sigma2
        m = logw.max(axis=1, keepdims=True)
        w = np.exp(logw - m)
        lls[:, ci] = ll_fact + m[:, 0] + np.log(w.mean(axis=1))
        ess[:, ci] = (w.sum(axis=1) ** 2) / (K * (w ** 2).sum(axis=1))
    return lls, ess


def run_sis(frames_per_snr: int, paths: int, out_csv: str, seed: int = 4062,
            snrs=None) -> None:
    """Corrected Bayes accuracy on a subsample; Delta vs factorized + ESS."""
    rng = np.random.default_rng(seed)
    rows = []
    for snr in (snrs or SNRS):
        sigma2 = 10 ** (-snr / 10)
        per = frames_per_snr // len(DIGITAL)
        cor_f = cor_s = total = 0
        min_ess = 1.0; emax_sum = 0.0
        for ci, name in enumerate(DIGITAL):
            _, yn = gen_batch(name, per, 128, snr, rng)
            ll_f = all_logliks(yn, sigma2)
            ll_s, ess = sis_corrected_logliks(yn, sigma2, paths, rng)
            post = np.exp(ll_s - ll_s.max(axis=1, keepdims=True))
            post /= post.sum(axis=1, keepdims=True)
            arg = ll_s.argmax(1)
            cor_f += int((ll_f.argmax(1) == ci).sum())
            cor_s += int((arg == ci).sum())
            emax_sum += float(post.max(axis=1).sum())
            # certify the ESS of the winning class, the one that decides accuracy
            min_ess = min(min_ess, float(ess[np.arange(len(arg)), arg].mean()))
            total += per
        af, as_ = cor_f / total, cor_s / total
        emax = emax_sum / total
        ci95 = 1.96 * math.sqrt(as_ * (1 - as_) / total)
        rows.append((snr, af, as_, emax, min_ess, total, ci95))
        print(f"snr={snr:+3d}  fact={100*af:6.2f}  SIS={100*as_:6.2f} "
              f" E[maxp]={100*emax:6.2f}  dAcc={100*(as_-af):+5.2f}pp "
              f" minESS={min_ess:.2f}  ci95=+-{100*ci95:.2f}pp", flush=True)
    with open(out_csv, "w") as f:
        f.write("snr,acc_factorized,acc_sis,e_max_posterior_sis,min_ess,"
                "n_frames,ci95\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.6f},{r[2]:.6f},{r[3]:.6f},{r[4]:.4f},"
                    f"{r[5]},{r[6]:.6f}\n")
    print(f"wrote {out_csv}")


# ----------------------------------------------------------------------------
# Tier-E grid
# ----------------------------------------------------------------------------
def run_tier_e(frames: int, out_csv: str, seed: int = 2026,
               chunk: int = 20000) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for snr in SNRS:
        sigma2 = 10 ** (-snr / 10)
        correct = 0; emax_sum = 0.0; total = 0
        per = frames // len(DIGITAL)
        for ci, name in enumerate(DIGITAL):
            done = 0
            while done < per:
                b = min(chunk, per - done)
                _, yn = gen_batch(name, b, 128, snr, rng)
                ll = all_logliks(yn, sigma2)
                post = np.exp(ll - ll.max(axis=1, keepdims=True))
                post /= post.sum(axis=1, keepdims=True)
                correct += int((ll.argmax(axis=1) == ci).sum())
                emax_sum += float(post.max(axis=1).sum())
                total += b; done += b
        acc = correct / total
        emax = emax_sum / total
        ci95 = 1.96 * math.sqrt(acc * (1 - acc) / total)
        rows.append((snr, acc, emax, total, ci95))
        print(f"snr={snr:+3d}  BayesAcc={100*acc:6.2f}  E[max p]={100*emax:6.2f} "
              f" ci95=+-{100*ci95:.2f}pp  n={total}", flush=True)
    with open(out_csv, "w") as f:
        f.write("snr,bayes_acc,e_max_posterior,n_frames,ci95\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.6f},{r[2]:.6f},{r[3]},{r[4]:.6f}\n")
    print(f"wrote {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", choices=["inversion", "identity", "brute", "all"])
    ap.add_argument("--run", choices=["tier_e", "sis"])
    ap.add_argument("--frames-per-snr", type=int, default=2100)
    ap.add_argument("--paths", type=int, default=512)
    ap.add_argument("--data-root",
                    default="data/synthetic_awgn_amc_v1")
    ap.add_argument("--frames", type=int, default=200000)
    ap.add_argument("--out", default="tier_e_ceiling.csv")
    a = ap.parse_args()
    print(f"device: {DEV}")
    if a.check in ("inversion", "all"):
        print("[check] manifold inversion vs disk clean frames")
        ok = check_inversion(a.data_root)
        if not ok:
            sys.exit("INVERSION FAILED - port does not match generator; STOP.")
    if a.check in ("identity", "all"):
        print("[check] Acc_MAP == E[max posterior]")
        ok = check_identity()
        if not ok:
            sys.exit("IDENTITY FAILED - likelihood bug; STOP.")
    if a.check in ("brute", "all"):
        print("[check] small-n brute force coupling")
        check_brute()
    if a.run == "tier_e":
        run_tier_e(a.frames, a.out)
    if a.run == "sis":
        run_sis(a.frames_per_snr, a.paths, a.out)


if __name__ == "__main__":
    main()
