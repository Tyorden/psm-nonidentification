"""
psm_core.py — Shared Van Westendorp Price Sensitivity Meter (PSM) implementation.

Replicates the intersection logic of the R package `pricesensitivitymeter`
(Alletsee) / the classic Van Westendorp (1976) construction, in numpy.

Per-respondent data: four price thresholds
    tc  = "too cheap"        (below this, price seems too cheap / low quality)
    ch  = "cheap / a bargain"
    ex  = "expensive / getting pricey"
    te  = "too expensive"    (at/above this, will not buy — too costly)
with the within-respondent order tc <= ch <= ex <= te.

The four PSM curves are functionals of the FOUR MARGINAL distributions only:
    toocheap(p)     = P(tc >= p)   [descending]   share who find p "too cheap"
    cheap(p)        = P(ch >= p)   [descending]   share who find p "cheap"
    expensive(p)    = P(ex <= p)   [ascending]    share who find p "expensive"
    tooexpensive(p) = P(te <= p)   [ascending]    share who find p "too expensive"

The four reported price points (all functionals of the marginals only):
    OPP = "Optimal Price Point"          : toocheap  == tooexpensive
    IPP = "Indifference Price Point"      : cheap     == expensive
    PMC = "Point of Marginal Cheapness"   : toocheap  == expensive     (lower bound)
    PME = "Point of Marginal Expensiveness": tooexpensive == cheap      (upper bound)

These four definitions match `pricesensitivitymeter::psm_analysis` ($opp, $idp,
$pricing_range = c(pmc, pme)).

This module is deliberately dependency-light (numpy only).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Share (survival / cdf) curves — functionals of the marginals only.
# Implemented with searchsorted (O(N log N + |grid|), tiny memory) rather than
# an outer broadcast, so populations of millions x fine grids are cheap.
# ---------------------------------------------------------------------------
def _survival(x, grid):
    """P(X >= p) for each p in grid."""
    xs = np.sort(np.asarray(x, dtype=float))
    n = xs.size
    # count of xs >= p  =  n - (#{xs < p})  =  n - searchsorted(xs, p, 'left')
    return (n - np.searchsorted(xs, grid, side="left")) / n


def _cdf(x, grid):
    """P(X <= p) for each p in grid."""
    xs = np.sort(np.asarray(x, dtype=float))
    n = xs.size
    # count of xs <= p  =  searchsorted(xs, p, 'right')
    return np.searchsorted(xs, grid, side="right") / n


def share_toocheap(tc, grid):
    return _survival(tc, grid)


def share_cheap(ch, grid):
    return _survival(ch, grid)


def share_expensive(ex, grid):
    return _cdf(ex, grid)


def share_tooexpensive(te, grid):
    return _cdf(te, grid)


# ---------------------------------------------------------------------------
# Intersection finder: crossing of two curves sampled on a sorted grid.
# Linear interpolation of the crossing price, matching the R package's
# interpolate=TRUE behaviour. Returns (price, height) or None if no crossing.
# ---------------------------------------------------------------------------
def _first_crossing(grid, f, g):
    """First price where f - g changes sign; linear interpolation of the root."""
    d = np.asarray(f) - np.asarray(g)
    # exact zero on the grid
    for i in range(len(grid)):
        if d[i] == 0.0:
            return float(grid[i]), float(f[i])
    sign = np.sign(d)
    idx = np.where(np.diff(sign) != 0)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    x0, x1 = grid[i], grid[i + 1]
    d0, d1 = d[i], d[i + 1]
    t = d0 / (d0 - d1)  # d0 + t*(d1-d0) = 0
    price = x0 + t * (x1 - x0)
    height = f[i] + t * (f[i + 1] - f[i])
    return float(price), float(height)


def psm_points(tc, ch, ex, te, grid=None, n_grid=4001):
    """Compute the four Van Westendorp price points from respondent thresholds."""
    tc, ch, ex, te = map(np.asarray, (tc, ch, ex, te))
    if grid is None:
        lo = min(tc.min(), ch.min(), ex.min(), te.min())
        hi = max(tc.max(), ch.max(), ex.max(), te.max())
        pad = 0.02 * (hi - lo + 1e-9)
        grid = np.linspace(lo - pad, hi + pad, n_grid)
    grid = np.asarray(grid, dtype=float)

    tcv = share_toocheap(tc, grid)
    chv = share_cheap(ch, grid)
    exv = share_expensive(ex, grid)
    tev = share_tooexpensive(te, grid)

    return {
        "grid": grid,
        "toocheap": tcv, "cheap": chv, "expensive": exv, "tooexpensive": tev,
        "OPP": _first_crossing(grid, tcv, tev),   # toocheap  vs tooexpensive
        "IPP": _first_crossing(grid, chv, exv),   # cheap     vs expensive
        "PMC": _first_crossing(grid, tcv, exv),   # toocheap  vs expensive
        "PME": _first_crossing(grid, tev, chv),   # tooexpensive vs cheap
    }


# ---------------------------------------------------------------------------
# Demand / revenue helpers.
# ---------------------------------------------------------------------------
def demand_from_wtp(wtp, grid):
    """Empirical demand D(p) = P(WTP >= p)."""
    return _survival(wtp, grid)


def revenue_optimum(wtp, grid):
    """Argmax_p p * P(WTP >= p) on the grid. Returns (p_star, revenue, D)."""
    D = demand_from_wtp(wtp, grid)
    R = grid * D
    i = int(np.argmax(R))
    return float(grid[i]), float(R[i]), D


def fmt_pt(pt):
    return "None" if pt is None else f"p={pt[0]:.4f} (height={pt[1]:.4f})"
