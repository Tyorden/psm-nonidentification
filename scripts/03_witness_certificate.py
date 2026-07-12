"""
03_witness_certificate.py — FULL-SUPPORT CERTIFICATE for the discrete witness.

Closes the referee nit found in the Fable verification pass
(notes/VERIFICATION_FABLE.md section 4, item 2): scripts/01 evaluated the
revenue candidates on a grid capped at 30 while the witness population's
support extends to max(te) = 44. The hand check showed the argmax is
unaffected, but the published certificate must cover the whole support by
construction. This script re-certifies the witness on a grid past the full
support AND by exact atom-level enumeration (no grid at all), so the revenue
argmax claim is grid-free.

Also emits the numbers used by the paper's positive corollaries:
  - Corollary 2 (measurement equation restores identification): the identified
    optimum under the reading W = te is p = 22 and under W = ch is p = 12,
    both away from OPP = 14 -> even under a declared reading, the OPP crossing
    is not the identified revenue maximizer.
  - Corollary 1 worked example (degenerate band, W ~ Uniform(2,3)): PSM then
    identifies the demand curve exactly; OPP = median = 2.5 while
    p* = 2 (revenue 2.0 vs 1.25 at OPP -> 37.5% loss). Identification of the
    demand curve does not rescue the OPP reading.

Outputs: results/03_witness_certificate.json, results/03_curves_full.csv
"""
import importlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from psm_core import fmt_pt, psm_points, revenue_optimum

wit = importlib.import_module("01_nonidentification")  # TYPES, expand()

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

# Full-support grid: population spans [4, 44]; grid runs to 50.
GRID = np.linspace(0.0, 50.0, 50001)


def atom_revenue_table(types, col):
    """Exact revenue p * P(W >= p) at every atom of the WTP law (grid-free).

    For a discrete WTP law, p * P(W >= p) is linear and increasing in p
    between consecutive atoms (P(W >= p) is constant on each (a_k, a_{k+1}]),
    so the supremum over all p > 0 is attained at an atom. Enumerating atoms
    is therefore an exact global optimization over the whole support.
    """
    atoms = sorted({t[col] for t in types})
    w = np.array([t[1] for t in types])
    vals = np.array([t[col] for t in types], dtype=float)
    table = []
    for a in atoms:
        share = float(w[vals >= a].sum())
        table.append({"price": a, "P(W>=p)": round(share, 10),
                      "revenue": round(a * share, 10)})
    best = max(table, key=lambda r: r["revenue"])
    return table, best


def rev_at_generic(types, col, p):
    """Exact revenue p * P(W >= p) for the type mixture at an arbitrary p."""
    w = np.array([t[1] for t in types])
    vals = np.array([t[col] for t in types], dtype=float)
    return p * float(w[vals >= p].sum())


def main():
    print("=" * 72)
    print("FULL-SUPPORT WITNESS CERTIFICATE (grid to 50, exact atom check)")
    print("=" * 72)

    tc, ch, ex, te, wa, wb = wit.expand()
    okA, okB = wit.check_coherence(tc, te, wa, wb)
    assert okA and okB

    psm = psm_points(tc, ch, ex, te, grid=GRID)
    OPP = psm["OPP"][0]
    print("\nPSM points on the full-support grid [0, 50]:")
    for k in ("PMC", "OPP", "IPP", "PME"):
        print(f"    {k}: {fmt_pt(psm[k])}")

    # grid optima over the full support
    pA, rA, DA = revenue_optimum(wa, GRID)
    pB, rB, DB = revenue_optimum(wb, GRID)

    # exact atom-level optima (grid-free): W_A is column 6? use TYPES layout
    # TYPES rows: (name, weight, tc, ch, ex, te, W_A, W_B)
    tabA, bestA = atom_revenue_table(wit.TYPES, 6)
    tabB, bestB = atom_revenue_table(wit.TYPES, 7)

    print("\nExact atom-level revenue, reading A (W = ch):")
    for r in tabA:
        print(f"    p = {r['price']:>4}   P(W>=p) = {r['P(W>=p)']:.2f}   "
              f"revenue = {r['revenue']:.2f}")
    print("\nExact atom-level revenue, reading B (W = te):")
    for r in tabB:
        print(f"    p = {r['price']:>4}   P(W>=p) = {r['P(W>=p)']:.2f}   "
              f"revenue = {r['revenue']:.2f}")

    print(f"\n    p*_A = {bestA['price']} (exact)  vs grid {pA:.4f}")
    print(f"    p*_B = {bestB['price']} (exact)  vs grid {pB:.4f}")
    assert abs(pA - bestA["price"]) < 1e-9, "grid and atom argmax must agree"
    assert abs(pB - bestB["price"]) < 1e-9, "grid and atom argmax must agree"
    straddle = bestA["price"] < OPP < bestB["price"]
    print(f"    straddle: {bestA['price']} < OPP {OPP:.4f} < {bestB['price']}"
          f"  ->  {'YES' if straddle else 'NO'}")
    assert straddle

    # revenue lost at OPP (exact, using atom shares)
    lossA = 1 - rev_at_generic(wit.TYPES, 6, OPP) / bestA["revenue"]
    lossB = 1 - rev_at_generic(wit.TYPES, 7, OPP) / bestB["revenue"]
    print(f"    revenue loss at OPP:  A = {100*lossA:.1f}%   B = {100*lossB:.1f}%")

    # ---- tighter band check: readings inside [ch, ex] also straddle --------
    # A referee may object that any sensible reading lives in the inner band
    # ch <= W <= ex rather than [tc, te]. The readings W = ch and W = ex are
    # both coherent with the inner band, and their optima straddle OPP too.
    tabCH, bestCH = atom_revenue_table(wit.TYPES, 3)   # W = ch (same as A)
    tabEX, bestEX = atom_revenue_table(wit.TYPES, 4)   # W = ex
    lossEX = 1 - rev_at_generic(wit.TYPES, 4, OPP) / bestEX["revenue"]
    print("\nInner-band readings (ch <= W <= ex):")
    print(f"    p*(W=ch) = {bestCH['price']}   p*(W=ex) = {bestEX['price']}"
          f"   OPP = {OPP:.4f}")
    inner_straddle = bestCH["price"] < OPP < bestEX["price"]
    print(f"    inner-band straddle -> {'YES' if inner_straddle else 'NO'}"
          f"   (loss at OPP under W=ex: {100*lossEX:.1f}%)")
    assert inner_straddle

    # ---- Corollary 2 numbers: identified optimum under a declared reading ---
    # Under W = te the demand law is the observed too-expensive marginal, and
    # the identified optimum is exactly p*_B; under W = ch it is p*_A.
    corollary2 = {
        "identified_optimum_if_W_equals_te": bestB["price"],
        "identified_optimum_if_W_equals_ch": bestA["price"],
        "OPP": OPP,
        "note": "even with the reading declared (identification restored), "
                "the OPP crossing is not the identified revenue maximizer",
    }

    # ---- Corollary 1 worked example: degenerate band, W ~ Uniform(2,3) ------
    # F_tc = F_te = F_W. OPP solves 1 - F = F -> median = 2.5. Demand
    # D(p) = min(1, max(0, 3 - p)) on [2,3]; R(p) = p D(p); R'(p) = 3 - 2p < 0
    # on [2,3], so p* = 2 with R = 2. R(2.5) = 1.25 -> 37.5% loss at OPP.
    g = np.linspace(1.0, 3.0, 20001)
    D = np.clip(3.0 - g, 0.0, 1.0)
    R = g * D
    i = int(np.argmax(R))
    corollary1_example = {
        "W_law": "Uniform(2,3), tc = te = W (degenerate band)",
        "OPP_equals_median": 2.5,
        "p_star_grid": float(g[i]),
        "revenue_at_p_star": float(R[i]),
        "revenue_at_OPP": float(2.5 * (3.0 - 2.5)),
        "loss_at_OPP": 1 - (2.5 * 0.5) / float(R[i]),
    }
    print("\nCorollary 1 example (W ~ U(2,3), band degenerate):")
    print(f"    OPP = median = 2.5 ; p* = {g[i]:.4f} ; "
          f"loss at OPP = {100*corollary1_example['loss_at_OPP']:.1f}%")

    out = {
        "grid": {"lo": 0.0, "hi": 50.0, "n": len(GRID)},
        "population_support": {"min": 4, "max": 44},
        "coherence": {"A_tc<=W<=te": okA, "B_tc<=W<=te": okB},
        "psm_points_full_grid": {
            k: {"price": psm[k][0], "height": psm[k][1]}
            for k in ("PMC", "OPP", "IPP", "PME")
        },
        "atom_revenue_table_A": tabA,
        "atom_revenue_table_B": tabB,
        "p_star_A_exact": bestA["price"],
        "revenue_A_exact": bestA["revenue"],
        "p_star_B_exact": bestB["price"],
        "revenue_B_exact": bestB["revenue"],
        "p_star_A_grid": pA,
        "p_star_B_grid": pB,
        "straddles_OPP": bool(straddle),
        "psm_error_A_signed_exact": 14 - bestA["price"],
        "psm_error_B_signed_exact": 14 - bestB["price"],
        "revenue_loss_at_OPP_A": lossA,
        "revenue_loss_at_OPP_B": lossB,
        "inner_band_check": {
            "constraint": "ch <= W <= ex",
            "atom_revenue_table_ex": tabEX,
            "p_star_W_eq_ch": bestCH["price"],
            "p_star_W_eq_ex": bestEX["price"],
            "straddles_OPP": bool(inner_straddle),
            "revenue_loss_at_OPP_W_eq_ex": lossEX,
        },
        "corollary2_measurement_equation": corollary2,
        "corollary1_degenerate_band_example": corollary1_example,
    }
    with open(os.path.join(RESULTS, "03_witness_certificate.json"), "w") as f:
        json.dump(out, f, indent=2)

    header = "price,toocheap,cheap,expensive,tooexpensive,demand_A,demand_B,rev_A,rev_B"
    stride = 50
    rows = np.column_stack([
        GRID, psm["toocheap"], psm["cheap"], psm["expensive"],
        psm["tooexpensive"], DA, DB, GRID * DA, GRID * DB,
    ])[::stride]
    np.savetxt(os.path.join(RESULTS, "03_curves_full.csv"), rows,
               delimiter=",", header=header, comments="", fmt="%.6f")

    print("\nwrote results/03_witness_certificate.json, results/03_curves_full.csv")
    print("\nCERTIFICATE: full-support grid and exact atom enumeration agree; "
          "witness stands.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
