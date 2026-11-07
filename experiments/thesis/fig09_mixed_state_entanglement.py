"""
Thesis Figure 9: the mixed-state subtlety. Under noise the subsystem entropy
rises while the entanglement falls (Phase 4, experiment A).

    python experiments\\thesis\\fig09_mixed_state_entanglement.py

Panels (N = CFG.thesis.noise_size, p = N/2, h = CFG.thesis.noise_field;
HVA solid, HEA dashed; colours are the three channels)
  (a) half-chain entropy S(rho_A): grows with noise (mixing, not entanglement)
  (b) logarithmic negativity E_N = ln ||rho^T_A||_1: falls, reaching 0
      (entanglement sudden death)
  (c) conditional entropy S(A|B) = S(rho_AB) - S(rho_B): negative values
      certify distillable entanglement (hashing bound); it crosses 0

Validation (all 18 files, 3 channels, 11 strengths, 20 fields)
  * Pure states at zero noise: S(rho_AB) = 0, purity 1, S_A = S_B,
    I = 2 S_A and I_2 = 2 S2_A at every cut, S(A|B) = -S_A, and
    E_N = S_1/2 >= S_A (Renyi ordering).
  * Every state: S_A >= S2_A >= 0, Araki-Lieb S(rho_AB) >= |S_A - S_B|,
    I >= 0, E_N >= 0, S_A <= (N/2) ln 2.
  * E_N decreases monotonically with the strength for every state with
    E_N(0) >= CFG.thesis.negativity_threshold (all channels).
  * Depolarizing: S_A rises and E_N falls between zero and the strongest noise
    at every such state.
  * Sudden death (E_N = 0 at h_main) and hashing-bound strengths reproduce the
    Phase 4 report.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from config import CFG
from experiments.thesis.figlib import FigureSpec, field_index, run_figure
from utils.plotting import (CHANNEL_COLORS, CHANNEL_LABELS, COLORS, label_zero_strength,
                            panel_labels, reference_line, thesis_figure, zero_strength_positions)

SPEC = FigureSpec(
    number=9,
    slug="mixed_state_entanglement",
    chapter="Noise robustness",
    phases=(4,),
    title="Subsystem entropy rises while entanglement falls under noise",
    caption=("Entanglement measures of the noisy N = 8, p = 4 states at h = J against the noise "
             "strength, for the three channels (colours), HVA (solid) and HEA (dashed). (a) The "
             "half-chain entropy grows with noise although (b) the logarithmic negativity falls, "
             "reaching zero (entanglement sudden death) for depolarizing noise at p = 0.05. "
             "(c) The conditional entropy S(A|B) turns positive, so the hashing bound no longer "
             "certifies distillable entanglement. Strength 0 is drawn at the left edge."),
)

ANSATZE = (("hva", "HVA", "-"), ("hea", "HEA", "--"))
LN2 = np.log(2.0)


def first_crossing(s, y, level=0.0, log_x=True) -> float:
    """First strength where y crosses `level` upward, interpolated (log strength)."""
    for k in range(len(s) - 1):
        if y[k] < level <= y[k + 1] and s[k] > 0:
            x0, x1 = (np.log(s[k]), np.log(s[k + 1])) if log_x else (s[k], s[k + 1])
            x = x0 + (level - y[k]) * (x1 - x0) / (y[k + 1] - y[k])
            return float(np.exp(x) if log_x else x)
    return float("nan")


def build(ctx):
    th, nz = CFG.thesis, CFG.noise
    sizes = list(CFG.system.sizes_noisy)
    keys = ("S_A", "S_B", "S2_A", "S_total", "purity", "I", "I2", "E_N", "S_cond", "S_A_profile",
            "S2_A_profile", "I_profile", "I2_profile", "strengths", "channels", "h", "n_1q", "n_2q")
    files = {(a, N, p): ctx.load(f"noisy_eval_{a}_N{N}_p{p}", "phase4", require=keys)
             for a, *_ in ANSATZE for N in sizes for p in CFG.system.depths(N, noisy=True)}
    report = ctx.load_json("phase4_report_data.json", "phase4")
    first = next(iter(files.values()))
    channels = [str(c) for c in first["channels"]]
    s = np.asarray(first["strengths"], dtype=float)
    h = np.asarray(first["h"])
    Np, ih = th.noise_size, field_index(h, th.noise_field)
    thr, tol = th.negativity_threshold, 1e-9
    ctx.param(plot_size=Np, plot_depth=Np // 2, field=th.noise_field, channels=channels,
              negativity_threshold=thr)

    # -- validation ------------------------------------------------------------
    pure = []
    order, araki, mi, en, cap = [], [], [], [], []
    mono, rises, falls, n_ent = [], True, True, 0
    for (a, N, p), d in files.items():
        z = lambda k: np.asarray(d[k])[:, 0]                          # noqa: E731  strength 0
        pure.append(max(np.max(np.abs(z("S_total"))), np.max(np.abs(z("purity") - 1)),
                        np.max(np.abs(z("S_A") - z("S_B"))),
                        np.max(np.abs(z("I_profile") - 2 * z("S_A_profile"))),
                        np.max(np.abs(z("I2_profile") - 2 * z("S2_A_profile"))),
                        np.max(np.abs(z("S_cond") + z("S_A")))))
        order.append(np.max(z("S_A") - z("E_N")))
        araki.append(np.max(np.abs(np.asarray(d["S_A"]) - d["S_B"]) - d["S_total"]))
        order.append(np.max(np.asarray(d["S2_A"]) - d["S_A"]))
        mi.append(-np.min(d["I"]))
        en.append(-np.min(d["E_N"]))
        cap.append(np.max(d["S_A"]) - (N // 2) * LN2)
        E = np.asarray(d["E_N"])
        ent = E[:, 0, :] >= thr                                       # [channel, field]
        if ent.any():
            mono.append(np.max(np.where(ent[:, None, :], np.diff(E, axis=1), -np.inf)))
        c = channels.index("depolarizing")
        sel = ent[c]
        n_ent += int(sel.sum())
        rises &= bool(np.all(d["S_A"][c, -1, sel] > d["S_A"][c, 0, sel]))
        falls &= bool(np.all(E[c, -1, sel] < E[c, 0, sel]))
    ctx.check_max("pure states at zero noise: S_AB = 0, purity 1, S_A = S_B, I = 2S_A, I_2 = 2S2_A, "
                  "S(A|B) = -S_A", pure, tol)
    ctx.check_max("Renyi ordering: E_N(0) = S_1/2 >= S_A, and S_A >= S2_A for every state", order, tol)
    ctx.check_max("Araki-Lieb S(rho_AB) >= |S_A - S_B| (every state)", araki, tol)
    ctx.check_true("I >= 0, E_N >= 0, S_A <= (N/2) ln 2 (every state)",
                   max(mi) <= tol and max(en) <= tol and max(cap) <= tol,
                   f"min I {-max(mi):.1e}, min E_N {-max(en):.1e}, max S_A - (N/2)ln2 {max(cap):.1e}")
    ctx.check_max(f"E_N decreases monotonically where E_N(0) >= {thr:g} (all channels)", mono,
                  nz.monotone_tol)
    ctx.check_true("depolarizing: S_A rises and E_N falls at every entangled state", rises and falls,
                   f"{n_ent} states with E_N(0) >= {thr:g}")

    death, mine_h, rep_h = {}, [], []
    c = channels.index("depolarizing")
    for (a, N, p), d in files.items():
        E = np.asarray(d["E_N"])[c, :, ih]
        hit = np.flatnonzero(E <= th.zero_tol)
        death[f"{a}_N{N}_p{p}"] = float(s[hit[0]]) if hit.size else None
    rep_death = report.get("sudden_death", {}).get("depolarizing", {})
    same = lambda u, v: (u is None and v is None) or (u is not None and v is not None  # noqa: E731
                                                      and abs(u - v) <= 1e-9 * max(abs(v), 1.0))
    ctx.check_true("sudden death (E_N = 0 at h_main, depolarizing) reproduces the Phase 4 report",
                   bool(rep_death) and all(same(death.get(k), v) for k, v in rep_death.items()),
                   "E_N = 0 for " + ", ".join(f"{k} (p = {v:g})" for k, v in death.items() if v))
    for ch, entries in report.get("hashing", {}).items():
        k = channels.index(ch)
        for key, val in entries.items():
            a, nn, pp = key.split("_")
            d = files[a, int(nn[1:]), int(pp[1:])]
            mine = first_crossing(s, np.asarray(d["S_cond"])[k, :, ih])
            theirs = val.get("p") if isinstance(val, dict) else None
            if theirs is None or isinstance(theirs, str):
                continue
            mine_h.append(mine)
            rep_h.append(theirs)
    if not mine_h:
        ctx.check_true("Phase 4 report has hashing-bound entries", False)
    ctx.check_close("hashing bound lost (S(A|B) = 0 at h_main): strengths reproduce the Phase 4 "
                    "report (every channel, ansatz, N, p)", mine_h, rep_h, rtol=th.hashing_rtol,
                    detail="log-interpolated strength")
    for a, name, _ in ANSATZE:
        d = files[a, Np, Np // 2]
        n = int(d["n_1q"]) + int(d["n_2q"])
        for ch in channels:
            k = channels.index(ch)
            p0 = first_crossing(s, np.asarray(d["S_cond"])[k, :, ih])
            ctx.note(f"{name} N={Np} p={Np // 2} h={th.noise_field:g}, {ch}: S_A {d['S_A'][k, 0, ih]:.3f} -> "
                     f"{d['S_A'][k, -1, ih]:.3f}, E_N {d['E_N'][k, 0, ih]:.3f} -> {d['E_N'][k, -1, ih]:.3f}, "
                     f"hashing lost at p = {p0:.2e} (lambda = {p0 * n:.3f})")

    # -- figure --------------------------------------------------------------------
    x, x0 = zero_strength_positions(s)
    fig, axes = thesis_figure(ncols=3, aspect=0.36)
    quantities = (("S_A", r"$S(\rho_A)$, half chain"), ("E_N", r"$\mathcal{E}_N$, half chain"),
                  ("S_cond", r"$S(A|B)$"))
    for ax, (key, ylabel) in zip(axes, quantities):
        for a, name, ls in ANSATZE:
            d = files[a, Np, Np // 2]
            for ch in channels:
                k = channels.index(ch)
                ax.semilogx(x, np.asarray(d[key])[k, :, ih], ls, color=CHANNEL_COLORS[ch], lw=1.0,
                            marker="o" if a == "hva" else "s", ms=2.2,
                            mfc="none" if a == "hea" else None)
        ax.set_xlabel("noise strength $p$")
        ax.set_ylabel(ylabel)
        label_zero_strength(ax, x0)
    reference_line(axes[2], 0.0, color=COLORS["grey"], lw=0.6)
    axes[0].set_title(f"$N={Np}$, $p={Np // 2}$, $h=J$", loc="right")
    handles = [axes[0].plot([], [], color=CHANNEL_COLORS[ch], label=CHANNEL_LABELS[ch])[0]
               for ch in channels]
    axes[0].legend(handles=handles, loc="upper left")
    hs = [axes[1].plot([], [], "o-", color=COLORS["black"], ms=2.2, label="solid: HVA")[0],
          axes[1].plot([], [], "s--", color=COLORS["black"], ms=2.2, mfc="none", label="dashed: HEA")[0]]
    axes[1].legend(handles=hs, loc="lower left")
    panel_labels(axes)
    return fig


if __name__ == "__main__":
    raise SystemExit(run_figure(SPEC, build))
