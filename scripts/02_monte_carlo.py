"""
02_monte_carlo.py — Monte-Carlo confirmation with CONTINUOUS distributions.

Two purposes:
  (1) Show the counterexample is not an artifact of hand-picked discrete atoms:
      reproduce identical-PSM / opposite-optimum with smooth valuations and
      interior (non-atomic) revenue optima.
  (2) Show the non-identification is a statement about the MARGINAL LAW, not a
      data-identity trick: draw the two environments' thresholds INDEPENDENTLY
      from the SAME marginal law; the PSM curves and all four price points then
      coincide up to Monte-Carlo error, while the revenue optima diverge and
      straddle the OPP.

Population model (comonotone in a latent valuation v):
    v ~ 0.60 * Normal(10, 2) + 0.40 * Normal(24, 3)      (bimodal -> tc/te
                                                           marginals overlap,
                                                           OPP well-defined)
    tc = v - 4,  ch = v - 1,  ex = v + 3,  te = v + 8     (fixed offsets =>
                                                           tc < ch < ex < te)
Coherent WTP readings, both in each respondent's band [tc, te] = [v-4, v+8]:
    W_A = ch      = v - 1     (reservation price = "bargain" threshold)
    W_B = te - 1  = v + 7     (reservation price just below "too expensive")

A standard PSM analysis observes ONLY (tc, ch, ex, te); it cannot tell A from B.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from psm_core import fmt_pt, psm_points, revenue_optimum

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

GRID = np.linspace(0.0, 45.0, 45001)


def _save_figure(fig, stem):
    """Journal figure rule (JRPM, read 2026-11-07): black and white only; EPS
    preferred, PDF and TIFF acceptable. Writes a vector PDF and EPS, a 600 dpi
    PNG, and (if Pillow is present) a 600 dpi RGB TIFF without an alpha
    channel."""
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".eps")
    fig.savefig(stem + ".png", dpi=600, facecolor="white")
    try:
        from PIL import Image
        im = Image.open(stem + ".png").convert("RGB")
        im.save(stem + ".tif", dpi=(600, 600), compression="tiff_lzw")
    except Exception as e:  # noqa
        print("tiff skipped:", e)


_RC = {"font.size": 8, "axes.titlesize": 9, "legend.fontsize": 7,
       "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7}
_CURVE_STYLES = (("toocheap", "too cheap", "black", "-"),
                 ("cheap", "cheap", "black", "--"),
                 ("expensive", "expensive", "0.45", "-"),
                 ("tooexpensive", "too expensive", "0.45", "--"))


def draw_v(n, rng):
    comp = rng.random(n) < 0.60
    v = np.where(comp, rng.normal(10, 2, n), rng.normal(24, 3, n))
    return v


def thresholds(v):
    return v - 4, v - 1, v + 3, v + 8   # tc, ch, ex, te


def run():
    print("=" * 72)
    print("MONTE-CARLO CONFIRMATION (continuous valuations)")
    print("=" * 72)
    log = {}

    # ---- (1) shared-respondent version (PSM literally identical) -----------
    rng = np.random.default_rng(11)
    N = 500_000
    v = draw_v(N, rng)
    tc, ch, ex, te = thresholds(v)
    assert np.all((tc <= ch) & (ch <= ex) & (ex <= te))
    W_A = ch.copy()
    W_B = te - 1.0
    # coherence
    cohA = bool(np.all((tc <= W_A) & (W_A <= te)))
    cohB = bool(np.all((tc <= W_B) & (W_B <= te)))

    psm = psm_points(tc, ch, ex, te, grid=GRID)
    OPP = psm["OPP"][0]
    pA, rA, DA = revenue_optimum(W_A, GRID)
    pB, rB, DB = revenue_optimum(W_B, GRID)

    print(f"\n[1] Shared thresholds (PSM identical for A & B). N={N:,}")
    print(f"    coherence tc<=W<=te:  A={cohA}  B={cohB}")
    for k in ("PMC", "OPP", "IPP", "PME"):
        print(f"    {k}: {fmt_pt(psm[k])}")
    print(f"    p*_A = {pA:.3f} (rev {rA:.3f})   "
          f"p*_B = {pB:.3f} (rev {rB:.3f})   OPP = {OPP:.3f}")
    straddle1 = min(pA, pB) < OPP < max(pA, pB)
    print(f"    optima on opposite sides of OPP ?  "
          f"min={min(pA,pB):.2f} < OPP={OPP:.2f} < max={max(pA,pB):.2f}  -> "
          f"{'STRADDLES' if straddle1 else 'NO'}")
    print("    (note: bimodal valuations cause revenue-mode switching, so the "
          "high/low\n     WTP reading is not monotone in p* here — but both "
          "optima still\n     land on OPPOSITE sides of the single reported OPP.)")

    def rev(w, p):
        return p * (w >= p).mean()
    lossA = 1 - rev(W_A, OPP) / rA
    lossB = 1 - rev(W_B, OPP) / rB
    print(f"    revenue lost pricing at OPP:  A={100*lossA:.1f}%  B={100*lossB:.1f}%")

    log["shared"] = {
        "N": N, "coherence_A": cohA, "coherence_B": cohB,
        "psm": {k: {"price": psm[k][0], "height": psm[k][1]}
                for k in ("PMC", "OPP", "IPP", "PME")},
        "p_star_A": pA, "rev_A": rA, "p_star_B": pB, "rev_B": rB,
        "OPP": OPP, "straddles": bool(straddle1),
        "loss_at_OPP_A": lossA, "loss_at_OPP_B": lossB,
    }

    # ---- (2) independent draws from the SAME marginal law ------------------
    print(f"\n[2] Independent draws from identical marginal law "
          f"(shows it is a marginal-law result):")
    rngA = np.random.default_rng(101)
    rngB = np.random.default_rng(202)
    vA = draw_v(N, rngA)
    vB = draw_v(N, rngB)
    tcA, chA, exA, teA = thresholds(vA)
    tcB, chB, exB, teB = thresholds(vB)
    psmA = psm_points(tcA, chA, exA, teA, grid=GRID)
    psmB = psm_points(tcB, chB, exB, teB, grid=GRID)
    # max discrepancy in the four price points and in the raw curves
    dpp = {k: abs(psmA[k][0] - psmB[k][0]) for k in ("PMC", "OPP", "IPP", "PME")}
    curve_maxdiff = max(
        float(np.max(np.abs(psmA[c] - psmB[c])))
        for c in ("toocheap", "cheap", "expensive", "tooexpensive")
    )
    print(f"    max |curve_A - curve_B| over all four curves : {curve_maxdiff:.4f}")
    for k in ("PMC", "OPP", "IPP", "PME"):
        print(f"    |{k}_A - {k}_B| = {dpp[k]:.4f}   "
              f"(A={psmA[k][0]:.3f}, B={psmB[k][0]:.3f})")
    pA2, rA2, _ = revenue_optimum(chA, GRID)           # A reads WTP = ch
    pB2, rB2, _ = revenue_optimum(teB - 1.0, GRID)     # B reads WTP = te-1
    print(f"    but revenue optima diverge: p*_A={pA2:.3f}  p*_B={pB2:.3f}  "
          f"(OPP~{psmA['OPP'][0]:.3f})")
    log["independent"] = {
        "curve_max_abs_diff": curve_maxdiff,
        "pricepoint_abs_diff": dpp,
        "p_star_A": pA2, "p_star_B": pB2,
        "OPP_A": psmA["OPP"][0], "OPP_B": psmB["OPP"][0],
    }

    # ---- persist -----------------------------------------------------------
    with open(os.path.join(RESULTS, "02_monte_carlo.json"), "w") as f:
        json.dump(log, f, indent=2)
    print("\nwrote results/02_monte_carlo.json")

    # ---- optional figure (black and white, vector) --------------------------
    # Changed 2026-11-07 for the JRPM version of the paper: greyscale with
    # distinct line styles, larger fonts, 6.5 x 2.8 in, vector PDF/EPS plus a
    # 600 dpi TIFF without alpha (journal rule: black and white only). The
    # numerical outputs above are unchanged.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update(_RC)
        fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.8))
        g = GRID
        for key, lab, col, ls in _CURVE_STYLES:
            ax[0].plot(g, psm[key], color=col, ls=ls, lw=1.1, label=lab)
        for k in ("PMC", "OPP", "IPP", "PME"):
            ax[0].axvline(psm[k][0], ls="--" if k == "OPP" else ":", lw=0.8,
                          color="black" if k == "OPP" else "0.55")
        # PMC/IPP and OPP/PME lie within 0.7 price units of each other, so
        # each pair gets one two-line label beside its lines
        ax[0].text(psm["PMC"][0] - 0.5, 1.03,
                   f"PMC {psm['PMC'][0]:.2f}\nIPP {psm['IPP'][0]:.2f}",
                   fontsize=6.5, ha="right", va="bottom")
        ax[0].text(psm["PME"][0] + 0.5, 1.03,
                   f"OPP {psm['OPP'][0]:.2f}\nPME {psm['PME'][0]:.2f}",
                   fontsize=6.5, ha="left", va="bottom")
        ax[0].set_title("Four PSM curves (identical for A and B)")
        ax[0].set_xlabel("price"); ax[0].set_ylabel("share")
        ax[0].legend(loc="center right", bbox_to_anchor=(1.0, 0.40), frameon=False)
        ax[0].set_xlim(0, 45); ax[0].set_ylim(0, 1.25)
        ax[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])

        ax[1].plot(g, g * DA, color="black", ls="-", lw=1.1, label="revenue A (W = ch)")
        ax[1].plot(g, g * DB, color="0.45", ls="--", lw=1.1, label="revenue B (W = te - 1)")
        ax[1].axvline(OPP, color="black", ls="--", lw=0.8, label=f"OPP = {OPP:.1f}")
        ax[1].axvline(pA, color="black", ls=":", lw=1.2)
        ax[1].axvline(pB, color="0.45", ls=":", lw=1.2)
        ax[1].set_title("Revenue curves: optima straddle the OPP")
        ax[1].set_xlabel("price"); ax[1].set_ylabel("p * D(p)")
        ax[1].legend(loc="upper right", frameon=False); ax[1].set_xlim(0, 45)
        fig.tight_layout()
        _save_figure(fig, os.path.join(RESULTS, "02_figure"))
        print("wrote results/02_figure.pdf, .eps, .png, .tif")
    except Exception as e:  # noqa
        print("figure skipped:", e)

    ok = straddle1 and curve_maxdiff < 0.02
    print("\nRESULT:", "MC CONFIRMS non-identification." if ok else "check output.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
