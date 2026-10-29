"""
Phase 2 figures, numerical summary and validation checks, made only from saved data.

Run from the project root, after experiments\\phase2_vqe_sweep.py and
experiments\\phase2_barren_plateau.py:

    python experiments\\phase2_plots.py

Figures (figures/phase2/*.pdf and *.png):
    vqe_error_vs_depth     energy error and ground-space infidelity vs p at the anchor fields
    vqe_error_vs_field     energy error vs h at several depths, HVA and HEA (largest N)
    vqe_fidelity_vs_field  F (even sector) and F_GS vs h at p = N (largest N)
    vqe_required_depth     smallest p with relative error below CFG.vqe.target_rel_error
    vqe_warm_vs_cold       warm-started versus cold-started sweeps (errors, iterations)
    vqe_hva_depthscan      HVA error vs p / N up to p = 2N
    barren_plateau         gradient variance at random initialization vs N

Numbers and PASS / FAIL / NOTE lines go to results/phase2/phase2_summary.txt.
Only the sizes found on disk are used, so the script also works after --quick.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import CFG, RESULTS_DIR
from utils.io import load_results, result_exists
from utils.plotting import (COLORS, PALETTE, SERIES_COLORS, SINGLE_COLUMN, save_figure,
                            set_thesis_style)
from vqe.gradients import fit_exponential_decay, fit_power_law
from vqe.metrics import ed_reference, required_depth

SUBDIR = "phase2"
ANSATZE = ("hva", "hea")
LABEL = {"hva": "HVA", "hea": "HEA"}
MARKER = {"hva": "o", "hea": "s"}
LINESTYLE = {"hva": "-", "hea": "--"}
FLOOR = 1e-15          # plotted floor for errors that reach machine precision
SUMMARY = []
CHECKS = []


def note(line: str = "") -> None:
    print(line)
    SUMMARY.append(line)


def check(name: str, ok: bool, detail: str) -> None:
    status = "PASS" if ok else "FAIL"
    CHECKS.append((status, name))
    note(f"  [{status}] {name}: {detail}")


def observe(name: str, detail: str) -> None:
    CHECKS.append(("NOTE", name))
    note(f"  [NOTE] {name}: {detail}")


def floor(x):
    return np.maximum(np.asarray(x, dtype=float), FLOOR)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load(name: str):
    return load_results(name, SUBDIR)[0] if result_exists(name, SUBDIR) else None


def available_sizes(ansatz: str) -> list:
    sizes = [n for n in CFG.system.sizes_noiseless
             if all(result_exists(f"vqe_{ansatz}_N{n}_p{p}", SUBDIR) for p in CFG.system.depths(n))]
    return sizes


def sweep_table(ansatz: str, n: int) -> dict:
    """Stack the sweep files of all depths: arrays of shape (n_p, n_h)."""
    depths = CFG.system.depths(n)
    files = [load(f"vqe_{ansatz}_N{n}_p{p}") for p in depths]
    keys = ("energy", "E0", "rel_error", "fidelity", "fidelity_gs", "n_iter", "success", "parity")
    out = {k: np.array([f[k] for f in files]) for k in keys}
    out["h"] = files[0]["h"]
    out["depths"] = np.array(depths)
    return out


def symmetry_broken_floor(n: int, h: np.ndarray) -> np.ndarray:
    """
    Lowest relative error of a state with weight 1/2 in each parity sector:
    eps_SB = (E_odd - E_even) / (2 |E0|). A symmetry-broken HEA state cannot go below it.
    """
    se = ed_reference(n, h, verbose=False)["sector_energies"][:, :, 0]
    e0 = se.min(axis=1)
    return (se[:, 1] - se[:, 0]) / (2 * np.abs(e0))


def size_color(n: int) -> str:
    sizes = list(CFG.system.sizes_noiseless)
    return PALETTE[sizes.index(n) % len(PALETTE)] if n in sizes else COLORS["black"]


def field_index(h, value):
    return int(np.argmin(np.abs(h - value)))


def critical_line(ax):
    ax.axvline(CFG.tfim.h_critical, color="0.5", ls=":", lw=1.2)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_error_vs_depth(tables):
    fields = CFG.vqe.anchor_fields
    fig, axes = plt.subplots(2, len(fields), figsize=(SINGLE_COLUMN, SINGLE_COLUMN * 0.68),
                             sharex=True, sharey="row")
    for a in ANSATZE:
        for n, t in tables[a].items():
            for j, hv in enumerate(fields):
                i = field_index(t["h"], hv)
                kw = dict(color=size_color(n), ls=LINESTYLE[a], marker=MARKER[a], ms=3)
                axes[0, j].semilogy(t["depths"], floor(t["rel_error"][:, i]), **kw)
                axes[1, j].semilogy(t["depths"], floor(1 - t["fidelity_gs"][:, i]), **kw)
    for j, hv in enumerate(fields):
        axes[0, j].set_title(f"$h/J = {hv:g}$", fontsize=10)
        axes[1, j].set_xlabel("depth $p$")
    axes[0, 0].set_ylabel(r"$\epsilon = (E - E_0)/|E_0|$")
    axes[1, 0].set_ylabel(r"$1 - F_{\mathrm{GS}}$")
    handles = [plt.Line2D([], [], color="0.3", ls=LINESTYLE[a], marker=MARKER[a], ms=3,
                          label=LABEL[a]) for a in ANSATZE]
    sizes = sorted({n for a in ANSATZE for n in tables[a]})
    handles += [plt.Line2D([], [], color=size_color(n), label=f"$N={n}$") for n in sizes]
    axes[0, -1].legend(handles=handles, fontsize=7, loc="lower left")
    fig.tight_layout()
    save_figure(fig, "vqe_error_vs_depth", SUBDIR)
    plt.close(fig)


def fig_error_vs_field(tables):
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COLUMN, SINGLE_COLUMN * 0.42), sharey=True)
    for ax, a in zip(axes, ANSATZE):
        if not tables[a]:
            continue
        n = max(tables[a])
        t = tables[a][n]
        picks = sorted({1, max(n // 4, 1), n // 2, n})
        for k, p in enumerate(picks):
            ax.semilogy(t["h"], floor(t["rel_error"][p - 1]), color=PALETTE[k % len(PALETTE)],
                        label=f"$p={p}$")
        if a == "hea":
            ordered = t["h"] < CFG.tfim.h_critical
            ax.semilogy(t["h"][ordered], floor(symmetry_broken_floor(n, t["h"])[ordered]),
                        color="0.5", ls=":", lw=1.4, label=r"$\epsilon_{\mathrm{SB}}$")
        critical_line(ax)
        ax.set_xlabel("$h/J$")
        ax.set_title(f"{LABEL[a]}, $N={n}$", fontsize=10)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"$\epsilon$")
    fig.tight_layout()
    save_figure(fig, "vqe_error_vs_field", SUBDIR)
    plt.close(fig)


def fig_fidelity_vs_field(tables):
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN * 0.62, SINGLE_COLUMN * 0.45))
    for a in ANSATZE:
        if not tables[a]:
            continue
        n = max(tables[a])
        t = tables[a][n]
        c = SERIES_COLORS[LABEL[a]]
        ax.plot(t["h"], t["fidelity"][-1], color=c, ls="--", label=f"{LABEL[a]} $F$")
        ax.plot(t["h"], t["fidelity_gs"][-1], color=c, ls="-", label=rf"{LABEL[a]} $F_{{\mathrm{{GS}}}}$")
    critical_line(ax)
    ax.set_xlabel("$h/J$")
    ax.set_ylabel("fidelity")
    ax.set_title(f"$p = N = {max(max(tables['hva'], default=0), max(tables['hea'], default=0))}$",
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, "vqe_fidelity_vs_field", SUBDIR)
    plt.close(fig)


def required_depth_curves(tables):
    target = CFG.vqe.target_rel_error
    curves = {}
    for a in ANSATZE:
        for n, t in tables[a].items():
            curves[(a, n)] = np.array([required_depth(t["depths"], t["rel_error"][:, i], target)
                                       for i in range(len(t["h"]))])
    return curves


def fig_required_depth(tables, curves):
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COLUMN, SINGLE_COLUMN * 0.42), sharey=True)
    for ax, a in zip(axes, ANSATZE):
        for n, t in tables[a].items():
            pstar = curves[(a, n)]
            reached = np.isfinite(pstar)
            ax.plot(t["h"][reached], pstar[reached], "o-", ms=3, color=size_color(n), label=f"$N={n}$")
            ax.plot(t["h"][~reached], np.full((~reached).sum(), n + 0.5), "x", ms=4,
                    color=size_color(n))
        critical_line(ax)
        ax.set_xlabel("$h/J$")
        ax.set_title(LABEL[a], fontsize=10)
    axes[0].set_ylabel(rf"$p^*$  ($\epsilon < {CFG.vqe.target_rel_error:g}$)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, "vqe_required_depth", SUBDIR)
    plt.close(fig)


def fig_warm_vs_cold(warmcold):
    if not warmcold:
        return
    fig, axes = plt.subplots(2, len(warmcold), figsize=(SINGLE_COLUMN, SINGLE_COLUMN * 0.62),
                             sharex=True, squeeze=False)
    for j, (a, d) in enumerate(warmcold.items()):
        c = SERIES_COLORS[LABEL[a]]
        axes[0, j].semilogy(d["h"], floor(d["cold_rel_error"]), color="0.5", label="cold")
        axes[0, j].semilogy(d["h"], floor(d["warm_rel_error"]), color=c, label="warm")
        axes[1, j].plot(d["h"], d["cold_total_fev"] / d["cold_n_candidates"], color="0.5",
                        label="cold (per run)")
        axes[1, j].plot(d["h"], d["warm_n_fev"], color=c, label="warm")
        axes[0, j].set_title(f"{LABEL[a]}, $N={int(d['n_qubits'])}$, $p={int(d['depth'])}$",
                             fontsize=10)
        axes[1, j].set_xlabel("$h/J$")
        for ax in axes[:, j]:
            critical_line(ax)
    axes[0, 0].set_ylabel(r"$\epsilon$")
    axes[1, 0].set_ylabel("gradient evaluations")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, "vqe_warm_vs_cold", SUBDIR)
    plt.close(fig)


def fig_depthscan(scans):
    if not scans:
        return
    fields = CFG.vqe.anchor_fields
    fig, axes = plt.subplots(1, len(fields), figsize=(SINGLE_COLUMN, SINGLE_COLUMN * 0.4),
                             sharey=True)
    for n, d in scans.items():
        for j, ax in enumerate(axes):
            ax.semilogy(d["depths"] / n, floor(d["rel_error"][j]), "o-", ms=2.5,
                        color=size_color(n), label=f"$N={n}$")
    for j, ax in enumerate(axes):
        ax.axvline(1.0, color="0.5", ls=":", lw=1.2)
        ax.set_title(f"HVA, $h/J={fields[j]:g}$", fontsize=10)
        ax.set_xlabel("$p/N$")
    axes[0].set_ylabel(r"$\epsilon$")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, "vqe_hva_depthscan", SUBDIR)
    plt.close(fig)


BP_FAMILIES = {"hea_linear": ("HEA, $p=N$", SERIES_COLORS["HEA"], "-"),
               "hea_fixed": (f"HEA, $p={CFG.vqe.grad_var_fixed_depth}$", SERIES_COLORS["HEA"], "--"),
               "hva_linear": ("HVA, $p=N$", SERIES_COLORS["HVA"], "-")}


def fig_barren(bp):
    if not bp:
        return
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COLUMN, SINGLE_COLUMN * 0.42))
    titles = {"local": r"local cost $\langle Z_c Z_{c+1}\rangle$",
              "energy": rf"energy density, $h/J={CFG.vqe.grad_var_field:g}$"}
    for ax, cost in zip(axes, ("local", "energy")):
        for fam, d in bp.items():
            label, color, ls = BP_FAMILIES[fam]
            ax.semilogy(d["sizes"], d[f"{cost}_var_center"], marker="o", ms=3.5, color=color,
                        ls=ls, label=label)
        if "hea_linear" in bp:
            d = bp["hea_linear"]
            ref = d[f"{cost}_var_center"][0] * 2.0 ** (-(d["sizes"] - d["sizes"][0]))
            ax.semilogy(d["sizes"], ref, color="0.5", ls=":", label=r"$\propto 2^{-N}$")
        ax.set_xlabel("$N$")
        ax.set_title(titles[cost], fontsize=10)
    axes[0].set_ylabel(r"$\mathrm{Var}[\partial_\theta C]$")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, "barren_plateau", SUBDIR)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Validation checks and summary
# ---------------------------------------------------------------------------
def summarize_sweeps(tables, curves):
    note("Field sweeps (relative error eps, ground-space fidelity F_GS)")
    for a in ANSATZE:
        for n, t in tables[a].items():
            i_c = field_index(t["h"], CFG.tfim.h_critical)
            note(f"  {LABEL[a]} N={n:2d}: at p = N, max eps over h = {t['rel_error'][-1].max():.2e}, "
                 f"eps(h_c) = {t['rel_error'][-1, i_c]:.2e}, min F_GS = {t['fidelity_gs'][-1].min():.5f}; "
                 f"fraction of runs at max_iterations = {np.mean(t['n_iter'] >= CFG.vqe.max_iterations):.2f}")
    note()

    note("Validation checks")
    # 1. Variational principle
    worst = min((t["rel_error"].min() for a in ANSATZE for t in tables[a].values()), default=np.nan)
    check("variational principle", worst > -1e-10, f"min eps over all runs = {worst:.2e} (must be >= 0)")

    # 2. HVA parity conservation
    if tables["hva"]:
        dev = max(np.abs(t["parity"] - 1).max() for t in tables["hva"].values())
        check("HVA stays in the even-parity sector", dev < 1e-10, f"max |<P> - 1| = {dev:.1e}")

        # 3. HVA energy non-increasing in depth (identity-layer warm start)
        rise = max(np.max(np.diff(t["energy"], axis=0)) for t in tables["hva"].values())
        check("HVA energy non-increasing in depth", rise < 1e-9,
              f"max E(p) - E(p-1) = {rise:.1e}")

    # 4. Plan: N = 4 HVA reaches near-unit fidelity at modest depth
    if 4 in tables["hva"]:
        t = tables["hva"][4]
        worst_f = t["fidelity"].min(axis=1)
        p_good = required_depth(t["depths"], 1 - worst_f, 1e-3)
        check("N = 4 HVA near-unit fidelity", worst_f[-1] > 1 - 1e-6,
              f"worst-case F over all fields: " + ", ".join(
                  f"p={p}: {f:.6f}" for p, f in zip(t["depths"], worst_f))
              + f"; F > 0.999 everywhere from p = {p_good:g}")

    # 5. Plan: required depth increases near the critical point
    for a in ANSATZE:
        for n, t in tables[a].items():
            pstar = np.where(np.isfinite(curves[(a, n)]), curves[(a, n)], n + 1)
            h = t["h"]
            crit = pstar[np.abs(h - CFG.tfim.h_critical) <= 0.1].max()
            dis = pstar[h >= h.max() - 0.25].max()
            ordered = pstar[h <= h.min() + 0.25].max()
            text = (f"{LABEL[a]} N={n}: p* = {dis:g} (disordered, h >= {h.max() - 0.25:.2f}), "
                    f"{crit:g} (|h - h_c| <= 0.1), {ordered:g} (ordered, h <= {h.min() + 0.25:.2f}); "
                    f"N+1 means not reached")
            if n < 6:
                observe("required depth vs field", text)
            elif a == "hea":
                check("HEA: critical region needs the most depth", crit >= max(dis, ordered), text)
            else:
                check("HVA: critical region needs more depth than the paramagnet", crit >= dis, text)
                if ordered > crit:
                    observe("HVA: ordered phase is the hardest region", text +
                            ". From |+> the HVA must build the cat state; with shared angles on an "
                            "open chain this needs p > N (see vqe_hva_depthscan)")

    # 6. HEA symmetry breaking in the ordered phase
    for n, t in tables["hea"].items():
        if n < 6:
            continue
        broken = t["fidelity"][-1] < 0.6
        if broken.any():
            # rigorous bound E >= w_even E_even + w_odd E_odd, with w_odd = (1 - <P>) / 2
            w_odd = (1 - t["parity"][-1, broken]) / 2
            bound = 2 * w_odd * symmetry_broken_floor(n, t["h"])[broken]
            ratio = t["rel_error"][-1, broken] / bound
            check(f"HEA N={n} symmetry-broken states sit at the parity-splitting bound",
                  ratio.min() > 1 - 1e-6 and np.median(ratio) < 2,
                  f"{broken.sum()} fields with F < 0.6 at p = N; eps / bound from "
                  f"{ratio.min():.3f} to {ratio.max():.3f} (median {np.median(ratio):.3f}); "
                  f"bound = w_odd (E_odd - E_even) / |E0|, w_odd = (1 - <P>)/2")
        sel = t["h"] <= 0.5
        gap = (t["fidelity_gs"][-1, sel] - t["fidelity"][-1, sel]).mean()
        observe(f"HEA N={n} symmetry breaking", f"for h <= 0.5 at p = N: mean F = "
                f"{t['fidelity'][-1, sel].mean():.3f}, mean F_GS = {t['fidelity_gs'][-1, sel].mean():.4f} "
                f"(F_GS - F = {gap:.3f}; a symmetry-broken state has F = 1/2)")
    note()


def summarize_warmcold(warmcold):
    if not warmcold:
        return
    note("Warm start versus cold start (T2.5)")
    for a, d in warmcold.items():
        tol = CFG.vqe.target_rel_error
        it_cold = (d["cold_total_fev"] / d["cold_n_candidates"]).mean()
        it_warm = d["warm_n_fev"].mean()
        note(f"  {LABEL[a]} N={int(d['n_qubits'])} p={int(d['depth'])}: points with eps < {tol:g}: "
             f"cold {np.mean(d['cold_rel_error'] < tol):.2f}, warm {np.mean(d['warm_rel_error'] < tol):.2f}; "
             f"max eps cold {d['cold_rel_error'].max():.2e}, warm {d['warm_rel_error'].max():.2e}")
        note(f"      gradient evaluations per optimization: cold {it_cold:.0f}, warm {it_warm:.0f}; "
             f"total per sweep: cold {d['cold_total_fev'].sum()}, warm {d['warm_total_fev'].sum()}")
        conv_cold = np.mean(d["cold_rel_error"] < tol)
        conv_warm = np.mean(d["warm_rel_error"] < tol)
        speedup = it_cold / it_warm
        text = (f"{speedup:.1f}x fewer evaluations per optimization, "
                f"{d['cold_total_fev'].sum() / d['warm_total_fev'].sum():.1f}x less total work, "
                f"converged fraction {conv_warm:.2f} (warm) vs {conv_cold:.2f} (cold)")
        if a == "hva":
            check("HVA warm start speeds up convergence at equal accuracy",
                  speedup > 1.5 and conv_warm >= conv_cold - 0.05, text)
        else:
            observe("HEA warm start", text + ". Most HEA runs stop at max_iterations either way, "
                    "so warm starting alone is cheaper but less reliable; the main sweeps "
                    "therefore combine the warm start with random restarts")
    note()


def summarize_depthscan(scans, pbc):
    if pbc is not None:
        note("Periodic HVA at p = N/2 (known exact result, Ho & Hsieh 2019)")
        for n, row, frow in zip(pbc["sizes"], pbc["rel_error"], pbc["fidelity"]):
            note(f"  N={n:2d}: eps = " + ", ".join(f"{e:.1e}" for e in row)
                 + f"   min F = {frow.min():.12f}")
        check("periodic HVA exact at p = N/2", pbc["rel_error"].max() < 1e-9,
              f"max eps = {pbc['rel_error'].max():.1e}")
        note()
    if scans:
        thr = 1e-6
        note(f"Open-chain HVA depth scan: smallest p with eps < {thr:g}")
        for n, d in scans.items():
            parts = []
            for j, hv in enumerate(d["fields"]):
                pe = required_depth(d["depths"], d["rel_error"][j], thr)
                if np.isfinite(pe):
                    parts.append(f"h={hv:g}: p = {pe:g} ({pe / n:.2f} N)")
                else:
                    parts.append(f"h={hv:g}: not reached by p = {d['depths'][-1]} "
                                 f"(min eps = {d['rel_error'][j].min():.1e})")
            note(f"  N={n:2d}: " + ", ".join(parts))
        note()


def summarize_barren(bp):
    if not bp:
        return
    note("Barren plateaus (T2.7): fits of Var[d C / d theta] for the light-cone parameter")
    fits = {}
    for fam, d in bp.items():
        for cost in ("local", "energy"):
            e = fit_exponential_decay(d["sizes"], d[f"{cost}_var_center"])
            p = fit_power_law(d["sizes"], d[f"{cost}_var_center"])
            fits[(fam, cost)] = e
            note(f"  {fam:11s} {cost:6s}: exp(-b N) with b = {e['b']:.3f} +/- {e['b_err']:.3f};  "
                 f"N^(-k) with k = {p['k']:.2f} +/- {p['k_err']:.2f}   (N = {d['sizes'].min()}..{d['sizes'].max()})")
    if ("hea_linear", "local") in fits:
        b = fits[("hea_linear", "local")]["b"]
        check("HEA (p = N) decays like a 2-design", abs(b - np.log(2)) < 0.1,
              f"b = {b:.3f}, ln 2 = {np.log(2):.3f}")
    if ("hea_linear", "local") in fits and ("hva_linear", "local") in fits:
        bh, bv = fits[("hea_linear", "local")]["b"], fits[("hva_linear", "local")]["b"]
        check("HVA has no barren plateau", bv < bh / 3,
              f"HVA b = {bv:.3f} vs HEA b = {bh:.3f}; the HVA decay is polynomial (free-fermion algebra)")
    note()


def main():
    set_thesis_style()
    note("PHASE 2 SUMMARY: noiseless VQE for the open TFIM")
    note("=" * 60)
    tables = {a: {n: sweep_table(a, n) for n in available_sizes(a)} for a in ANSATZE}
    note(f"Sizes found: HVA {sorted(tables['hva'])}, HEA {sorted(tables['hea'])}")
    note()
    n_wc = CFG.vqe.warmcold_size
    warmcold = {a: d for a in ANSATZE
                if (d := load(f"vqe_warmcold_{a}_N{n_wc}_p{max(n_wc // 2, 1)}")) is not None}
    scans = {n: d for n in CFG.vqe.hva_depth_scan_sizes
             if (d := load(f"vqe_hva_depthscan_N{n}")) is not None}
    pbc = load("vqe_hva_pbc_check")
    bp = {fam: d for fam in BP_FAMILIES if (d := load(f"bp_{fam}")) is not None}

    curves = required_depth_curves(tables)
    if tables["hva"] or tables["hea"]:
        fig_error_vs_depth(tables)
        fig_error_vs_field(tables)
        fig_fidelity_vs_field(tables)
        fig_required_depth(tables, curves)
    fig_warm_vs_cold(warmcold)
    fig_depthscan(scans)
    fig_barren(bp)

    summarize_sweeps(tables, curves)
    summarize_warmcold(warmcold)
    summarize_depthscan(scans, pbc)
    summarize_barren(bp)

    n_fail = sum(s == "FAIL" for s, _ in CHECKS)
    n_pass = sum(s == "PASS" for s, _ in CHECKS)
    note(f"Checks: {n_pass} passed, {n_fail} failed, {sum(s == 'NOTE' for s, _ in CHECKS)} notes")
    out = RESULTS_DIR / SUBDIR / "phase2_summary.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(SUMMARY) + "\n", encoding="utf-8")
    print(f"[summary] written to {out}")


if __name__ == "__main__":
    main()
