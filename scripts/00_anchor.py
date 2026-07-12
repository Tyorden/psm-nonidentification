"""
00_anchor.py — CORRECTNESS ANCHOR for the PSM intersection implementation.

We verify psm_core against a worked example with *analytically known* Van
Westendorp intersection points, before layering any new analysis on top.

Analytic worked example (uniform marginals, comonotone respondents):
    tc ~ Uniform(0, 10)   F_tc(p) = p/10
    ch ~ Uniform(2, 12)   F_ch(p) = (p-2)/10
    ex ~ Uniform(4, 14)   F_ex(p) = (p-4)/10
    te ~ Uniform(6, 16)   F_te(p) = (p-6)/10
Comonotone coupling tc=10U, ch=2+10U, ex=4+10U, te=6+10U (U~Unif(0,1))
=> tc <= ch <= ex <= te a.s.  (valid PSM respondents)

Solving the four defining equations by hand (curves are linear on the overlap):
    OPP : 1 - p/10 = (p-6)/10        => p = 8      (height 0.2)
    IPP : 1 - (p-2)/10 = (p-4)/10    => p = 8      (height 0.4)
    PMC : 1 - p/10 = (p-4)/10        => p = 7      (height 0.3)
    PME : (p-6)/10 = 1 - (p-2)/10    => p = 9      (height 0.3)

Two checks:
  (1) Evaluate the four share curves on the EXACT analytic CDFs (no sampling)
      -> must recover 8, 8, 7, 9 to grid resolution.
  (2) Draw a large comonotone empirical sample -> must recover the same within
      Monte-Carlo tolerance.

Correspondence to R `pricesensitivitymeter`: the package builds the same four
survival/cdf step curves (too cheap & cheap descending; expensive & too
expensive ascending) and reports $opp (toocheap x tooexpensive), $idp
(cheap x expensive) and $pricing_range = c(PMC, PME) with PMC = toocheap x
expensive, PME = tooexpensive x cheap. With interpolate=TRUE it linearly
interpolates the crossing, exactly as _first_crossing does here.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from psm_core import (_first_crossing, demand_from_wtp, fmt_pt, psm_points,
                      share_cheap, share_expensive, share_toocheap,
                      share_tooexpensive)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

EXPECTED = {"OPP": 8.0, "IPP": 8.0, "PMC": 7.0, "PME": 9.0}
TOL_ANALYTIC = 1e-2   # grid resolution
TOL_SAMPLE = 5e-2     # Monte-Carlo


def analytic_check(n_grid=8001):
    grid = np.linspace(0.0, 16.0, n_grid)
    F_tc = np.clip(grid / 10.0, 0, 1)
    F_ch = np.clip((grid - 2) / 10.0, 0, 1)
    F_ex = np.clip((grid - 4) / 10.0, 0, 1)
    F_te = np.clip((grid - 6) / 10.0, 0, 1)
    tcv, chv = 1 - F_tc, 1 - F_ch          # survival (descending)
    exv, tev = F_ex, F_te                  # cdf (ascending)
    pts = {
        "OPP": _first_crossing(grid, tcv, tev),
        "IPP": _first_crossing(grid, chv, exv),
        "PMC": _first_crossing(grid, tcv, exv),
        "PME": _first_crossing(grid, tev, chv),
    }
    return pts


def sample_check(N=400_000, seed=0):
    rng = np.random.default_rng(seed)
    U = rng.random(N)
    tc, ch, ex, te = 10 * U, 2 + 10 * U, 4 + 10 * U, 6 + 10 * U
    assert np.all((tc <= ch) & (ch <= ex) & (ex <= te)), "ordering violated"
    out = psm_points(tc, ch, ex, te, grid=np.linspace(0, 16, 8001))
    return {k: out[k] for k in EXPECTED}


def main():
    print("=" * 70)
    print("CORRECTNESS ANCHOR — PSM intersection logic")
    print("=" * 70)

    log = {"expected": EXPECTED, "analytic": {}, "sample": {}, "pass": True}

    print("\n[1] Analytic-CDF check (exact marginals):")
    a = analytic_check()
    for k, exp in EXPECTED.items():
        got = a[k][0]
        ok = abs(got - exp) <= TOL_ANALYTIC
        log["analytic"][k] = got
        log["pass"] &= ok
        print(f"    {k}: expected {exp:.3f}  got {got:.5f}  {fmt_pt(a[k])}  "
              f"[{'PASS' if ok else 'FAIL'}]")

    print("\n[2] Large-sample empirical check (N=400k comonotone):")
    s = sample_check()
    for k, exp in EXPECTED.items():
        got = s[k][0]
        ok = abs(got - exp) <= TOL_SAMPLE
        log["sample"][k] = got
        log["pass"] &= ok
        print(f"    {k}: expected {exp:.3f}  got {got:.5f}  "
              f"[{'PASS' if ok else 'FAIL'}]")

    print("\nRESULT:", "ALL CHECKS PASS" if log["pass"] else "FAILURE")
    with open(os.path.join(RESULTS, "00_anchor.json"), "w") as f:
        json.dump(log, f, indent=2)
    print("wrote results/00_anchor.json")
    return 0 if log["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
