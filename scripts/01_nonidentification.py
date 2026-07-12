"""
01_nonidentification.py — THE CORE COUNTEREXAMPLE.

Constructs two demand environments A and B that
  (i)  have BYTE-IDENTICAL Van Westendorp inputs -> identical four PSM curves ->
       identical OPP / IPP / PMC / PME, yet
  (ii) have revenue-optimal prices on OPPOSITE sides of the reported OPP.

Mechanism (the non-identification):
  The four PSM curves, and hence all four reported price points, are functionals
  of the four MARGINAL threshold distributions ONLY. The revenue-optimal price
  is argmax_p p * P(WTP >= p), which depends on the WILLINGNESS-TO-PAY (WTP)
  distribution. PSM never measures WTP. Given a fixed set of PSM answers, EVERY
  WTP distribution respecting the respondent's own bounds  tc <= W <= te  is
  consistent with those answers. That coherent family is wide enough to place
  the true optimum on either side of the OPP. Therefore the map
        demand primitive  ->  PSM output
  is many-to-one, and OPP does not identify the revenue-optimal price.

Both environments A and B share the SAME respondents and the SAME four thresholds
(so the PSM data are literally identical). They differ ONLY in the unobserved WTP
W_i, which in both environments satisfies tc_i <= W_i <= te_i (fully coherent).

Discrete headline population (4 segments), designed by hand. Segments span a
wide range so the "too cheap" and "too expensive" MARGINALS OVERLAP — i.e. a
premium respondent's "too cheap" price exceeds a bargain respondent's "too
expensive" price. That overlap makes the OPP a genuine positive-height crossing
(not a degenerate zero-height artifact):

  seg   weight    tc   ch   ex   te      W_A = ch     W_B = te
  T1     0.35      4    7   10   14          7           14
  T2     0.25      8   12   16   22         12           22
  T3     0.25     12   18   24   32         18           32
  T4     0.15     16   24   32   44         24           44

  W_A reads each respondent's reservation price as their "cheap / bargain"
  threshold (conservative buyer, only pays up to a bargain price).
  W_B reads it as their "too expensive" threshold (pays up to just below the
  point they call it too costly). BOTH satisfy tc <= W <= te for every
  respondent, so both are coherent with the identical PSM answers. The PSM
  literature never fixes which threshold is the reservation price — that
  ambiguity is exactly the non-identification we exploit.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from psm_core import demand_from_wtp, fmt_pt, psm_points, revenue_optimum

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

# ---- population definition --------------------------------------------------
TYPES = [
    # name, weight, tc, ch, ex, te, W_A(=ch), W_B(=te)
    ("T1", 0.35,  4,  7, 10, 14,  7, 14),
    ("T2", 0.25,  8, 12, 16, 22, 12, 22),
    ("T3", 0.25, 12, 18, 24, 32, 18, 32),
    ("T4", 0.15, 16, 24, 32, 44, 24, 44),
]


def expand(n=100_000):
    """Expand the type mixture into an explicit respondent population."""
    counts = [int(round(w * n)) for _, w, *_ in TYPES]
    tc, ch, ex, te, wa, wb = [], [], [], [], [], []
    for (name, w, t, c, e, x, a, b), k in zip(TYPES, counts):
        tc += [t] * k; ch += [c] * k; ex += [e] * k; te += [x] * k
        wa += [a] * k; wb += [b] * k
    return (np.array(tc, float), np.array(ch, float), np.array(ex, float),
            np.array(te, float), np.array(wa, float), np.array(wb, float))


def check_coherence(tc, te, wa, wb):
    okA = np.all((tc <= wa) & (wa <= te))
    okB = np.all((tc <= wb) & (wb <= te))
    return bool(okA), bool(okB)


def main():
    print("=" * 72)
    print("CORE COUNTEREXAMPLE — identical PSM, opposite revenue optima")
    print("=" * 72)

    tc, ch, ex, te, wa, wb = expand()
    okA, okB = check_coherence(tc, te, wa, wb)
    print(f"\nCoherence tc<=W<=te :  A={okA}   B={okB}")
    assert okA and okB, "WTP must respect each respondent's own bounds"

    grid = np.linspace(0.0, 30.0, 30001)

    # PSM is a function of the thresholds ONLY -> identical for A and B.
    psm = psm_points(tc, ch, ex, te, grid=grid)
    print("\nVan Westendorp price points (identical for A and B — same thresholds):")
    for k in ("PMC", "OPP", "IPP", "PME"):
        print(f"    {k}: {fmt_pt(psm[k])}")
    OPP = psm["OPP"][0]

    # Revenue optima from the two (coherent) WTP distributions.
    pA, rA, DA = revenue_optimum(wa, grid)
    pB, rB, DB = revenue_optimum(wb, grid)
    print(f"\nRevenue-optimal price  (argmax p * P(WTP>=p)):")
    print(f"    Environment A (low reading) : p*_A = {pA:.4f}   revenue = {rA:.4f}")
    print(f"    Environment B (high reading): p*_B = {pB:.4f}   revenue = {rB:.4f}")
    print(f"    Reported PSM 'optimal'      : OPP  = {OPP:.4f}")

    straddle = pA < OPP < pB
    print(f"\n  p*_A < OPP < p*_B  ?  {pA:.3f} < {OPP:.3f} < {pB:.3f}  ->  "
          f"{'YES — STRADDLES' if straddle else 'NO'}")

    # Quantify the error PSM makes in each environment.
    errA = OPP - pA
    errB = OPP - pB
    print(f"\n  PSM error (OPP - true optimum):")
    print(f"    Environment A: {errA:+.3f}  (PSM over-prices by {errA:.3f})")
    print(f"    Environment B: {errB:+.3f}  (PSM under-prices by {-errB:.3f})")
    print("  => same PSM output, error changes SIGN. OPP is not identifying.")

    # Revenue actually left on the table if the firm follows OPP.
    def rev(w, p):
        return p * (np.asarray(w) >= p).mean()
    lossA = 1 - rev(wa, OPP) / rA
    lossB = 1 - rev(wb, OPP) / rB
    print(f"\n  Revenue lost by pricing at OPP instead of the true optimum:")
    print(f"    Environment A: {100*lossA:5.1f}%")
    print(f"    Environment B: {100*lossB:5.1f}%")

    # --- persist everything -------------------------------------------------
    out = {
        "population_types": [
            {"name": n, "weight": w, "tc": t, "ch": c, "ex": e, "te": x,
             "W_A": a, "W_B": b}
            for (n, w, t, c, e, x, a, b) in TYPES
        ],
        "coherence": {"A_tc<=W<=te": okA, "B_tc<=W<=te": okB},
        "psm_points_identical_AB": {
            k: {"price": psm[k][0], "height": psm[k][1]}
            for k in ("PMC", "OPP", "IPP", "PME")
        },
        "revenue_optimum_A": {"p_star": pA, "revenue": rA},
        "revenue_optimum_B": {"p_star": pB, "revenue": rB},
        "straddles_OPP": bool(straddle),
        "psm_error_A_signed": errA,
        "psm_error_B_signed": errB,
        "revenue_loss_at_OPP_A": lossA,
        "revenue_loss_at_OPP_B": lossB,
    }
    with open(os.path.join(RESULTS, "01_counterexample.json"), "w") as f:
        json.dump(out, f, indent=2)

    # curves for plotting / inspection
    header = "price,toocheap,cheap,expensive,tooexpensive,demand_A,demand_B,rev_A,rev_B"
    stride = 30
    rows = np.column_stack([
        grid, psm["toocheap"], psm["cheap"], psm["expensive"],
        psm["tooexpensive"], DA, DB, grid * DA, grid * DB,
    ])[::stride]
    np.savetxt(os.path.join(RESULTS, "01_curves.csv"), rows,
               delimiter=",", header=header, comments="", fmt="%.6f")

    print("\nwrote results/01_counterexample.json, results/01_curves.csv")
    print("\nHEADLINE:", "COUNTEREXAMPLE CONSTRUCTED — identical PSM, "
          "opposite-side optima." if straddle else "straddle not achieved.")
    return 0 if straddle else 1


if __name__ == "__main__":
    sys.exit(main())
