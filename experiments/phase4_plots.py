"""
Phase 4 figures, validation checks and summary, from the saved results only.

Usage (from the project root, after experiments\\phase4_noise.py):
    python experiments\\phase4_plots.py

Writes figures/phase4/*.pdf and results/phase4/phase4_summary.txt.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from config import CFG, RESULTS_DIR
from noise.data import _arrays
from utils.plotting import SERIES_COLORS, save_figure, set_thesis_style

SUBDIR = "phase4"
ANS = ("hva", "hea")
LABEL = {"hva": "HVA", "hea": "HEA"}
CH_LABEL = {"depolarizing": "depolarizing", "amplitude_damping": "amplitude damping",
            "dephasing": "dephasing"}
SIZE_ALPHA = {4: 0.35, 6: 0.6, 8: 1.0}
LINES: list[str] = []
N_PASS = N_FAIL = N_NOTE = 0


def log(msg=""):
    print(msg)
    LINES.append(msg)


def check(ok: bool, msg: str):
    global N_PASS, N_FAIL
    N_PASS += bool(ok)
    N_FAIL += not ok
    log(f"[{'PASS' if ok else 'FAIL'}] {msg}")


def note(msg: str):
    global N_NOTE
    N_NOTE += 1
    log(f"[NOTE] {msg}")


def exists(name):
    return (RESULTS_DIR / SUBDIR / f"{name}.npz").exists()


def load(name):
    a = _arrays(name, SUBDIR)
    return {k: np.asarray(a[k]) for k in (a.files if hasattr(a, "files") else a.keys())}


def load_eval():
    out = {}
    for N in CFG.system.sizes_noisy:
        for a in ANS:
            for p in CFG.system.depths(N, noisy=True):
                nm = f"noisy_eval_{a}_N{N}_p{p}"
                if exists(nm):
                    out[a, N, p] = load(nm)
    return out


def load_train():
    out = {}
    n_starts = 1 + CFG.seeds.n_seeds_noisy
    for N in CFG.system.sizes_noisy:
        for a in ANS:
            for h in CFG.noise.train_fields:
                runs = [f"noisy_train_{a}_N{N}_h{h:.2f}_s{s}" for s in range(n_starts)]
                if all(exists(r) for r in runs):
                    out[a, N, h] = [load(r) for r in runs]
    return out


def ih_of(d, h):
    return int(np.flatnonzero(np.isclose(d["h"], h))[0])


def loglog_slope(x, y):
    m = (x > 0) & (y > 0)
    return np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0] if m.sum() >= 2 else np.nan


def cc_fit(profile, N):
    """c from S(ell) = (c/6) ln[(2N/pi) sin(pi ell/N)] + s', window 2 <= ell <= N/2 (Phase 1)."""
    ell = np.arange(CFG.ent.cc_ell_min, N // 2 + 1)
    x = np.log(2 * N / np.pi * np.sin(np.pi * ell / N))
    return 6 * np.polyfit(x, profile[ell - 1], 1)[0]


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------
def validate_eval(E):
    log("== Experiment A: noiseless optimum evaluated under noise ==")
    if not E:
        check(False, "no noisy_eval files found")
        return
    rs = max(d["repro_state_err"].max() for d in E.values())
    dE = max(np.abs(d["energy"][:, 0] - d["phase2_energy"]).max() for d in E.values())
    dF = max(max(np.abs(d["fidelity"][:, 0] - d["phase2_fidelity"]).max(),
                  np.abs(d["fidelity_gs"][:, 0] - d["phase2_fidelity_gs"]).max()) for d in E.values())
    check(rs < 1e-12, f"zero noise reproduces the Phase 2 states: max |rho - psi psi^+| = {rs:.1e}")
    check(dE < 1e-10, f"zero noise reproduces the Phase 2 energies: max diff = {dE:.1e}")
    check(dF < 1e-10, f"zero noise reproduces the Phase 2 fidelities F and F_GS: max diff = {dF:.1e}")
    with3 = [d for d in E.values() if np.all(np.isfinite(d["phase3_S"]))]
    if with3:
        dS = max(max(np.abs(d["S_A"][:, 0] - d["phase3_S"]).max(),
                     np.abs(d["S2_A"][:, 0] - d["phase3_S2"]).max()) for d in with3)
        check(dS < 1e-10, f"zero noise reproduces the Phase 3 half-chain S and S2: max diff = {dS:.1e}")
    else:
        note("Phase 3 entropies not found; S(N/2) reproduction check skipped")
    tr = max(np.abs(d["trace"] - 1).max() for d in E.values())
    he = max(d["herm_err"].max() for d in E.values())
    me = min(d["min_eig"].min() for d in E.values())
    check(tr < 1e-12 and he < 1e-13 and me > -1e-12,
          f"valid density matrices: max |Tr - 1| = {tr:.1e}, max Hermiticity error = {he:.1e}, min eigenvalue = {me:.1e}")
    ve = min(d["rel_error"].min() for d in E.values())
    check(ve > -1e-12, f"variational principle Tr(rho H) >= E0 at every point: min eps = {ve:.2e}")
    ee = max(d["estimator_err"].max() for d in E.values())
    check(ee < 1e-10, f"measurement-group energy equals Tr(rho H): max diff = {ee:.1e}")

    # purity monotone in strength
    tol = CFG.noise.monotone_tol
    chans = list(next(iter(E.values()))["channels"])
    for ic, ch in enumerate(chans):
        worst = max(np.diff(d["purity"][ic], axis=0).max() for d in E.values())
        n_bad = sum(int((np.diff(d["purity"][ic], axis=0) > tol).sum()) for d in E.values())
        msg = f"purity decreases monotonically with {ch} strength: largest increase {worst:.1e}, violations {n_bad}"
        if ch == "amplitude_damping" and n_bad:
            msg += " (non-unital channel: purity can recover toward |0...0>, see the gamma = 1 test)"
        check(n_bad == 0, msg)

    # weak-noise linear response
    s = next(iter(E.values()))["strengths"]
    weak = (s > 0) & (s <= CFG.noise.weak_noise_max)
    for ic, ch in enumerate(chans):
        slopes = []
        for d in E.values():
            for ih in range(d["h"].size):
                for key in ("purity", "fidelity_ideal"):
                    y = 1 - d[key][ic, weak, ih]
                    if np.all(y > 1e-12):
                        slopes.append(loglog_slope(s[weak], y))
        slopes = np.array(slopes)
        dev = np.nanmax(np.abs(slopes - 1))
        check(dev < CFG.noise.slope_tol,
              f"weak-noise linear response ({ch}): log-log slopes of 1 - purity and 1 - F_ideal for "
              f"p <= {CFG.noise.weak_noise_max:g}: {np.nanmin(slopes):.3f} to {np.nanmax(slopes):.3f}")

    # mixed-state subtlety
    log("-- mixed-state subtlety (half chain, strongest noise vs zero noise) --")
    for ic, ch in enumerate(chans):
        up = dn_EN = dn_I = tot = 0
        for d in E.values():
            ent = d["E_N"][ic, 0] > 0.01            # entangled at zero noise
            tot += int(ent.sum())
            up += int((d["S_A"][ic, -1] > d["S_A"][ic, 0])[ent].sum())
            dn_EN += int((d["E_N"][ic, -1] < d["E_N"][ic, 0])[ent].sum())
            dn_I += int((d["I"][ic, -1] < d["I"][ic, 0])[ent].sum())
        check(dn_EN == tot, f"{ch}: log negativity falls at all {tot} entangled points ({dn_EN})")
        note(f"{ch}: mutual information falls at {dn_I} of {tot} entangled points (not required: noise "
             f"inside the circuit is followed by gates that cross the cut and turn it into classical "
             f"correlations; data processing forbids an increase only for noise after the last such gate)")
        if ch == "depolarizing":
            check(up == tot, f"{ch}: subsystem entropy S(A) rises at all {tot} entangled points ({up}) "
                             f"although entanglement falls")
        else:
            note(f"{ch}: S(A) rises at {up} of {tot} entangled points (not required: "
                 f"{'non-unital, drives toward |0...0>' if ch == 'amplitude_damping' else 'Z-dephasing leaves Z-diagonal weight unchanged'})")

    # shots
    log("-- finite shots --")
    zs, ratios = [], []
    for d in E.values():
        R = d["E_shots"].shape[-1]
        exact = d["energy"][..., None]
        mean = d["E_shots"].mean(-1)
        zs.append(((mean - exact) / (d["E_sigma"] / np.sqrt(R))).ravel())
        ratios.append((d["E_shots"].std(-1, ddof=1) / d["E_sigma"]).ravel())
    zs, ratios = np.concatenate(zs), np.concatenate(ratios)
    frac = np.mean(np.abs(zs) > 3)
    check(frac < 0.01, f"shot estimates unbiased: {frac:.2%} of {zs.size} means beyond 3 sigma (Gaussian: 0.27%)")
    med = np.median(ratios)
    check(abs(med - 1) < 0.05, f"shot spread matches the analytic variance: median std/sigma = {med:.3f}")


def validate_train(T, E):
    log("== Experiment B: SPSA training under noise ==")
    if not T:
        note("no complete noisy_train sets found; training checks skipped")
        return
    gaps, floor = [], []
    for (a, N, h), runs in T.items():
        warm, cold = runs[0], runs[1:]
        gaps.append(warm["rel_error"] - warm["evalA_rel_error"])
        floor.append((a, N, h, np.median([c["rel_error"][0, 0] for c in cold])))
    gaps = np.concatenate([g.ravel() for g in gaps])
    check(np.median(gaps) < 1e-3,
          f"warm-started training under noise is not worse than the noiseless optimum under noise: "
          f"median eps_trained - eps_A = {np.median(gaps):.1e} (max {gaps.max():.1e})")
    note(f"noise-aware training improves on experiment A at {np.mean(gaps < 0):.0%} of points")
    for a, N, h, f in floor:
        if N == max(CFG.system.sizes_noisy):
            note(f"cold SPSA floor at zero noise ({LABEL[a]}, N = {N}, h = {h}): median eps = {f:.2e}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
METRICS = [("rel_error", r"$\epsilon$", True), ("fidelity_gs", r"$1 - F_{\mathrm{GS}}$", True),
           ("purity", r"purity Tr$\,\rho^2$", False), ("S2_A", r"$S_2(\rho_A)$", False),
           ("I", r"$I(A\!:\!B)$", False), ("E_N", r"$\mathcal{E}_N$", False)]


def fig_degradation(E, ch, h=1.0):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True)
    for (a, N, p), d in E.items():
        if p != N // 2:
            continue
        ic = list(d["channels"]).index(ch)
        ih = ih_of(d, h)
        s = d["strengths"]
        for ax, (key, lab, one_minus) in zip(axes.ravel(), METRICS):
            y = d[key][ic, :, ih]
            y = 1 - y if one_minus and key != "rel_error" else y
            ax.plot(s[1:], y[1:], "o-" if N == 8 else "-", ms=3, color=SERIES_COLORS[LABEL[a]],
                    alpha=SIZE_ALPHA[N], label=f"{LABEL[a]} N={N}")
            ax.axhline(y[0], color=SERIES_COLORS[LABEL[a]], alpha=0.3 * SIZE_ALPHA[N], ls=":", lw=0.8)
            ax.set_xscale("log")
            if one_minus:
                ax.set_yscale("log")
            ax.set_ylabel(lab)
    for ax in axes[1]:
        ax.set_xlabel(f"{CH_LABEL[ch]} strength $p$")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Degradation under {CH_LABEL[ch]} noise, $p = N/2$, $h = {h}$ (dotted: noiseless)")
    fig.tight_layout()
    save_figure(fig, f"noise_degradation_{ch}", SUBDIR)
    plt.close(fig)


def fig_subtlety(E, h_values=(0.5, 1.0)):
    N = max(CFG.system.sizes_noisy)
    chans = CFG.noise.channels
    fig, axes = plt.subplots(len(h_values), len(chans), figsize=(11, 3.2 * len(h_values)), sharex=True)
    axes = np.atleast_2d(axes)
    for r, h in enumerate(h_values):
        for c, ch in enumerate(chans):
            ax = axes[r, c]
            for a, ls in (("hva", "-"), ("hea", "--")):
                d = E.get((a, N, N // 2))
                if d is None:
                    continue
                ic, ih, s = list(d["channels"]).index(ch), ih_of(d, h), d["strengths"]
                for key, lab, col in (("S_A", r"$S(\rho_A)$", "C0"), ("I", r"$I/2$", "C1"),
                                      ("E_N", r"$\mathcal{E}_N$", "C2"), ("S_cond", r"$S(A|B)$", "C3")):
                    y = d[key][ic, 1:, ih] * (0.5 if key == "I" else 1)
                    ax.plot(s[1:], y, ls, color=col, label=f"{lab} {LABEL[a]}" if c == 0 else None)
            ax.axhline(0, color="k", lw=0.5)
            ax.axhline(np.log(2), color="grey", lw=0.5, ls=":")
            ax.set_xscale("log")
            ax.set_title(f"{CH_LABEL[ch]}, $h = {h}$", fontsize=9)
            if r == len(h_values) - 1:
                ax.set_xlabel("noise strength $p$")
    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.suptitle(f"Mixed-state subtlety, N = {N}, p = N/2: entropy rises, entanglement falls "
                 f"(solid HVA, dashed HEA)")
    fig.tight_layout()
    save_figure(fig, "noise_mixed_state_subtlety", SUBDIR)
    plt.close(fig)


def fig_shots(E, h=1.0, ch="depolarizing"):
    N = max(CFG.system.sizes_noisy)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, a in zip(axes, ANS):
        d = E.get((a, N, N // 2))
        if d is None:
            continue
        ic, ih, s = list(d["channels"]).index(ch), ih_of(d, h), d["strengths"]
        E0 = d["E0"][ih]
        ax.plot(s[1:], d["rel_error"][ic, 1:, ih], "k-", label="exact Tr$(\\rho H)$")
        for k, M in enumerate(d["shots"]):
            eps = (d["E_shots"][ic, 1:, ih, k] - E0) / abs(E0)
            ax.errorbar(s[1:] * (1 + 0.08 * k), eps.mean(-1), yerr=eps.std(-1), fmt="o", ms=3,
                        capsize=2, label=f"{M} shots (mean $\\pm$ 1 sd)")
            ax.axhline(d["E_sigma"][ic, 0, ih, k] / abs(E0), ls=":", lw=0.8, color=f"C{k}")
        ax.set_xscale("log")
        ax.set_title(f"{LABEL[a]}, N = {N}, p = {N // 2}, h = {h}", fontsize=9)
        ax.set_xlabel(f"{CH_LABEL[ch]} strength $p$")
    axes[0].set_ylabel(r"$\epsilon$ (dotted: single-run shot floor $\sigma/|E_0|$)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, "noise_shot_noise", SUBDIR)
    plt.close(fig)


def fig_training(T, h=1.0):
    sizes = [n for (_, n, hh) in T if np.isclose(hh, h)]
    if not sizes:
        return
    N = max(sizes)
    chans = CFG.noise.channels
    fig, axes = plt.subplots(len(chans), 2, figsize=(10, 3.0 * len(chans)), sharex=True)
    for c, a in enumerate(ANS):
        runs = T.get((a, N, h))
        if runs is None:
            continue
        s = runs[0]["strengths"]
        for r, ch in enumerate(chans):
            ax = axes[r, c]
            ic = list(runs[0]["channels"]).index(ch)
            ax.plot(s[1:], runs[0]["evalA_rel_error"][ic, 1:], "k-", label="A: noiseless optimum under noise")
            ax.plot(s[1:], runs[0]["rel_error"][ic, 1:], "o-", ms=3, color=SERIES_COLORS[LABEL[a]],
                    label="B: warm start, trained under noise")
            cold = np.array([x["rel_error"][ic, 1:] for x in runs[1:]])
            ax.plot(s[1:], np.median(cold, 0), "s--", ms=3, color=SERIES_COLORS["noisy"],
                    label="B: cold start (median)")
            ax.fill_between(s[1:], cold.min(0), cold.max(0), color=SERIES_COLORS["noisy"], alpha=0.2)
            ax.axhline(np.median([x["rel_error"][0, 0] for x in runs[1:]]), color=SERIES_COLORS["noisy"],
                       ls=":", lw=0.8)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{LABEL[a]}, {CH_LABEL[ch]}", fontsize=9)
            if c == 0:
                ax.set_ylabel(r"$\epsilon$")
            if r == len(chans) - 1:
                ax.set_xlabel("noise strength $p$")
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(f"Training under noise, N = {N}, p = N/2, h = {h}, "
                 f"{CFG.noise.train_shots} shots per basis (dotted: cold floor at zero noise)")
    fig.tight_layout()
    save_figure(fig, "noise_training", SUBDIR)
    plt.close(fig)


def fig_error_budget(E, h=1.0, ch="depolarizing"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for (a, N, p), d in E.items():
        ic, ih, s = list(d["channels"]).index(ch), ih_of(d, h), d["strengths"]
        lam = s[1:] * (d["n_1q"] + d["n_2q"])
        kw = dict(color=SERIES_COLORS[LABEL[a]], alpha=0.25 + 0.75 * p / (N // 2), lw=0.8 + 0.3 * p)
        axes[0].plot(s[1:], 1 - d["fidelity_ideal"][ic, 1:, ih], **kw)
        axes[1].plot(lam, 1 - d["fidelity_ideal"][ic, 1:, ih], **kw,
                     label=f"{LABEL[a]} N={N} p={p}" if N == 8 else None)
    lam = np.logspace(-3, 1, 50)
    axes[1].plot(lam, 1 - np.exp(-lam), "k:", label=r"$1 - e^{-\lambda}$")
    for ax, xl in zip(axes, ("noise strength $p$", r"error budget $\lambda = p\,(n_{1q} + n_{2q})$")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xl)
    axes[0].set_ylabel(r"$1 - \langle\psi_0|\rho|\psi_0\rangle$")
    axes[1].legend(fontsize=6, ncol=2)
    fig.suptitle(f"Infidelity to the noiseless state, {CH_LABEL[ch]}, h = {h}, all N and p")
    fig.tight_layout()
    save_figure(fig, "noise_error_budget", SUBDIR)
    plt.close(fig)


def fig_field_dependence(E, ch="depolarizing"):
    N = max(CFG.system.sizes_noisy)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for c, a in enumerate(ANS):
        d = E.get((a, N, N // 2))
        if d is None:
            continue
        ic, s = list(d["channels"]).index(ch), d["strengths"]
        cmap = plt.get_cmap("viridis")
        for k, js in enumerate(range(0, s.size, 2)):
            col = cmap(k / max(1, (s.size - 1) // 2))
            axes[0, c].plot(d["h"], d["S_A"][ic, js], color=col, label=f"p = {s[js]:.0e}" if js else "p = 0")
            axes[1, c].plot(d["h"], d["E_N"][ic, js], color=col)
        axes[0, c].set_title(f"{LABEL[a]}, N = {N}, p = {N // 2}, {CH_LABEL[ch]}", fontsize=9)
        axes[1, c].set_xlabel("$h$")
        for ax in axes[:, c]:
            ax.axvline(CFG.tfim.h_critical, color="grey", lw=0.5, ls=":")
    axes[0, 0].set_ylabel(r"$S(\rho_A)$")
    axes[1, 0].set_ylabel(r"$\mathcal{E}_N$")
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    save_figure(fig, "noise_field_dependence", SUBDIR)
    plt.close(fig)


def fig_central_charge(E, ch="depolarizing"):
    """Secondary quantity (Phase 3 decision): c at p = N/2 only, from S and from I/2."""
    N = max(CFG.system.sizes_noisy)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for a in ANS:
        d = E.get((a, N, N // 2))
        if d is None:
            continue
        ic, ih, s = list(d["channels"]).index(ch), ih_of(d, CFG.ent.cc_field), d["strengths"]
        cS = [cc_fit(d["S_A_profile"][ic, j, ih], N) for j in range(s.size)]
        cI = [cc_fit(0.5 * d["I_profile"][ic, j, ih], N) for j in range(s.size)]
        ax.plot(s[1:], cS[1:], "o-", ms=3, color=SERIES_COLORS[LABEL[a]], label=f"{LABEL[a]}: from $S(\\ell)$")
        ax.plot(s[1:], cI[1:], "s--", ms=3, color=SERIES_COLORS[LABEL[a]], label=f"{LABEL[a]}: from $I(\\ell)/2$")
        ax.axhline(cS[0], color=SERIES_COLORS[LABEL[a]], ls=":", lw=0.8)
        log(f"central charge {LABEL[a]} N={N} p={N // 2} h={CFG.ent.cc_field} ({ch}): c(0) = {cS[0]:.3f}; "
            f"p = {s[-1]:.0e}: from S {cS[-1]:.3f}, from I/2 {cI[-1]:.3f}")
    ax.axhline(0.5, color="k", lw=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(f"{CH_LABEL[ch]} strength $p$")
    ax.set_ylabel("fitted $c$ (window $2 \\leq \\ell \\leq N/2$)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, "noise_central_charge", SUBDIR)
    plt.close(fig)


# ---------------------------------------------------------------------------
def summary(E, T):
    log("== Key numbers: N = 8, p = N/2, h = h_c ==")
    N = max(CFG.system.sizes_noisy)
    for a in ANS:
        d = E.get((a, N, N // 2))
        if d is None:
            continue
        ih, s = ih_of(d, CFG.tfim.h_critical), d["strengths"]
        log(f"{LABEL[a]}: gates n_1q = {int(d['n_1q'])}, n_2q = {int(d['n_2q'])}; noiseless eps = "
            f"{d['rel_error'][0, 0, ih]:.2e}, F_GS = {d['fidelity_gs'][0, 0, ih]:.4f}, "
            f"S(A) = {d['S_A'][0, 0, ih]:.4f}, E_N = {d['E_N'][0, 0, ih]:.4f}")
        for ic, ch in enumerate(d["channels"]):
            for target in (1e-3, 1e-2, s[-1]):
                j = int(np.argmin(np.abs(s - target)))
                log(f"   {ch:>17s} p={s[j]:.1e}: eps={d['rel_error'][ic, j, ih]:.3e} "
                    f"1-F_GS={1 - d['fidelity_gs'][ic, j, ih]:.3e} purity={d['purity'][ic, j, ih]:.4f} "
                    f"S2(A)={d['S2_A'][ic, j, ih]:.4f} I={d['I'][ic, j, ih]:.4f} E_N={d['E_N'][ic, j, ih]:.4f}")
            neg = np.flatnonzero(d["S_cond"][ic, :, ih] > 0)
            log(f"   {ch:>17s}: S(A|B) turns positive (hashing bound lost) at p = "
                f"{s[neg[0]]:.1e}" if neg.size else f"   {ch:>17s}: S(A|B) < 0 up to p = {s[-1]:.0e}")


def main():
    set_thesis_style()
    E, T = load_eval(), load_train()
    log(f"Phase 4: {len(E)} eval files, {len(T)} complete training sets")
    validate_eval(E)
    validate_train(T, E)
    if E:
        for ch in CFG.noise.channels:
            fig_degradation(E, ch)
        fig_subtlety(E)
        fig_shots(E)
        fig_error_budget(E)
        fig_field_dependence(E)
        fig_central_charge(E)
        summary(E, T)
    if T:
        fig_training(T)
    log(f"\n{N_PASS} checks passed, {N_FAIL} failed, {N_NOTE} notes")
    (RESULTS_DIR / SUBDIR).mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / SUBDIR / "phase4_summary.txt").write_text("\n".join(LINES), encoding="utf-8")


if __name__ == "__main__":
    main()
