"""
04_empirical_range.py — EMPIRICAL DEMONSTRATION of the identified-range
recommendation (Section 8 of the paper) on data the author did not construct.

The paper proves the OPP does not identify the revenue-optimal price and
recommends reporting instead the IDENTIFIED RANGE [p*_ch, p*_te]: the
revenue-maximizing price when every respondent's willingness to pay W is read
as their "cheap/bargain" threshold (the most conservative coherent reading on
the inner band) versus as their "too expensive" threshold (the most generous
coherent reading). This script computes that range, the OPP a standard PSM
analysis would have reported, and the revenue cost of pricing at the OPP under
each endpoint reading, mirroring the paper's 28.2% / 2.1% construction
(results/03_witness_certificate.json), on two datasets:

DATASET 1 — REAL SURVEY DATA (primary).
    Dmitriev, Zenkova & Zenkov (2019), "Statistical properties of price
    sensitivity meter results", Tomsk State University Journal of Control and
    Computer Science 47, 30-40, doi:10.17223/19988605/47/4 — the same article
    cited as \\cite{dmitriev2019} in the paper. Their Table 1 publishes the
    complete respondent-level answers of a real PSM survey: representatives of
    N = 52 organizations, price preferences for a new B2B software product
    after a two-month trial. To protect commercial confidentiality the authors
    published the prices rescaled into conventional units ("u.e./unit"); a
    common multiplicative rescaling leaves every ratio and every percentage
    computed here unchanged. The file data/dmitriev2019_psm_n52.csv is a
    verbatim transcription (see its header for full provenance); this script
    certifies the transcription by reproducing all four price-point estimates
    published in the article's Table 2 (205 / 290 / 306 / 379) with the
    authors' own estimator (the (N+1)-th order statistic of the pooled
    2N-value sample, their eq. (8)).
    CAVEAT (recorded in the results file): the published table lists each of
    the four columns in ascending order, so the within-respondent pairing is
    not recoverable. Every quantity computed here — the four curves, the four
    price points, and the per-reading revenue optima p* = argmax p*P(W>=p)
    with W read as one threshold — is a functional of the four MARGINAL
    distributions alone (Lemma 1 of the paper), so the pairing does not
    affect any reported number. Under the as-printed row pairing (the
    comonotone coupling) all 52 rows satisfy ORD; 0 respondents are dropped.

DATASET 2 — SIMULATED DATA (secondary; the reference implementation's own
    example). The R package `pricesensitivitymeter` (Alletsee; the paper's
    \\cite{alletsee2025}) ships NO survey dataset: its repository
    (github.com/alletsee/pricesensitivitymeter, checked 2026-07-12) contains
    no data/ directory, and every example in its documentation and vignettes
    is explicitly simulated. We therefore use the package's canonical example
    exactly as published — the vignette "Visualizing PSM Results"
    (vignettes/visualizing-psm-results.Rmd) draws, after set.seed(20),
        toocheap     = round(rnorm(250, mean =  8, sd = 1.5), 2)
        cheap        = round(rnorm(250, mean = 12, sd = 2  ), 2)
        expensive    = round(rnorm(250, mean = 13, sd = 1  ), 2)
        tooexpensive = round(rnorm(250, mean = 15, sd = 1  ), 2)
    — and we reproduce that draw BIT-EXACTLY in pure Python by replicating
    R's default RNG (Mersenne-Twister seeded by R's set.seed scrambling, with
    rnorm's inversion method and the AS 241 normal quantile, which is also
    what CPython's statistics.NormalDist.inv_cdf implements). The replication
    is verified inside this script against two independently known R outputs
    (set.seed(1); rnorm(5) and set.seed(42); runif(2)). THIS DATASET IS
    SIMULATED and is labeled as such everywhere; it demonstrates the range
    computation on the example every user of the reference implementation
    sees, and it exercises the ORD-violation handling (independent draws
    produce many inconsistent respondents, which the paper's Section 2 —
    like the reference implementation — drops).

ORD handling (matches draft.tex Section 2): "A consistent respondent
satisfies tc <= ch <= ex <= te (ORD). Standard practice, and the reference
implementation below, drops respondents violating (ORD)." We drop violators
of the weak inequality and report how many were dropped. For transparency we
also report the count under the reference R package's stricter rule
(tc < ch < ex < te; psm_analysis validate=TRUE), which drops ties as well.

Everything is recomputed and asserted in-script: curve monotonicity, curve
bounds, cone-compatibility of the marginals, exact-rational atom enumeration
of every revenue optimum cross-checked against the grid maximizer, demand
recomputed two independent ways, and (for dataset 1) the published Table-2
price points reproduced exactly.

Outputs: results/04_empirical_range.json and the figure
         results/04_empirical_range.{pdf,eps,png,tif} (black and white
         since 2026-11-07; see the figure block).
Run:     python3 scripts/04_empirical_range.py   (numpy required; matplotlib
         optional for the figure; everything else is stdlib)
"""
import csv
import hashlib
import json
import os
import sys
from fractions import Fraction
from statistics import NormalDist

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from psm_core import fmt_pt, psm_points, _first_crossing  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(DATA, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

ACCESS_DATE = "2026-07-12"

# sha256 of the shipped transcription of Dmitriev et al. (2019), Table 1.
DMITRIEV_CSV = os.path.join(DATA, "dmitriev2019_psm_n52.csv")
DMITRIEV_SHA256 = "45c4edca85aecd669d816bc3e81c0e5f91edf4a89cf9564316d8c13fddd96337"
# Price points published in the article's Table 2 (their estimator: the
# (N+1)-th order statistic of the pooled sample of the two crossing columns).
DMITRIEV_TABLE2 = {"x_min(X1,X2)": 205, "x_opt(X1,X4)": 290,
                   "x_bezr(X3,X2)": 306, "x_max(X3,X4)": 379}


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


# ===========================================================================
# Exact replication of R's default RNG (Mersenne-Twister + inversion rnorm),
# needed to reproduce the pricesensitivitymeter vignette's simulated data.
# ===========================================================================
_M32 = 0xFFFFFFFF
_SND = NormalDist()


class RRng:
    """R's set.seed / unif_rand / norm_rand (RNG.c, snorm.c of R >= 1.7)."""
    N, M = 624, 397
    MATRIX_A, UPPER, LOWER = 0x9908B0DF, 0x80000000, 0x7FFFFFFF

    def __init__(self, seed):
        seed &= _M32
        for _ in range(50):                      # R's initial scrambling
            seed = (69069 * seed + 1) & _M32
        iseed = []
        for _ in range(625):
            seed = (69069 * seed + 1) & _M32
            iseed.append(seed)
        self.mt = iseed[1:]                      # i_seed[1..624] = MT state
        self.mti = 624                           # FixupSeeds: initial -> 624

    def _genrand_int(self):
        mt, N, M = self.mt, self.N, self.M
        if self.mti >= N:
            mag01 = (0, self.MATRIX_A)
            for kk in range(N - M):
                y = (mt[kk] & self.UPPER) | (mt[kk + 1] & self.LOWER)
                mt[kk] = mt[kk + M] ^ (y >> 1) ^ mag01[y & 1]
            for kk in range(N - M, N - 1):
                y = (mt[kk] & self.UPPER) | (mt[kk + 1] & self.LOWER)
                mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 1]
            y = (mt[N - 1] & self.UPPER) | (mt[0] & self.LOWER)
            mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ mag01[y & 1]
            self.mti = 0
        y = mt[self.mti]
        self.mti += 1
        y ^= y >> 11                             # MT tempering
        y = (y ^ ((y << 7) & 0x9D2C5680)) & _M32
        y = (y ^ ((y << 15) & 0xEFC60000)) & _M32
        y ^= y >> 18
        return y

    def unif_rand(self):
        x = self._genrand_int() * 2.3283064365386963e-10   # y / 2^32
        i2_32m1 = 2.328306437080797e-10                    # 1 / (2^32 - 1)
        if x <= 0.0:                                       # R's fixup()
            return 0.5 * i2_32m1
        if 1.0 - x <= 0.0:
            return 1.0 - 0.5 * i2_32m1
        return x

    def norm_rand(self):
        """R's inversion method: two uniforms -> AS241 quantile."""
        BIG = 134217728.0                                  # 2^27
        u1 = self.unif_rand()
        u1 = float(int(BIG * u1)) + self.unif_rand()
        return _SND.inv_cdf(u1 / BIG)

    def rnorm(self, n, mean=0.0, sd=1.0):
        return [mean + sd * self.norm_rand() for _ in range(n)]

    def runif(self, n):
        return [self.unif_rand() for _ in range(n)]


def verify_r_rng():
    """Certify the replication against independently known R outputs."""
    got = RRng(1).rnorm(5)
    exp = [-0.6264538, 0.1836433, -0.8356286, 1.5952808, 0.3295078]
    assert all(abs(a - b) < 5e-8 for a, b in zip(got, exp)), \
        f"set.seed(1); rnorm(5) mismatch: {got}"
    got = RRng(42).runif(2)
    exp = [0.9148060, 0.9370754]
    assert all(abs(a - b) < 5e-8 for a, b in zip(got, exp)), \
        f"set.seed(42); runif(2) mismatch: {got}"
    return {"rnorm_seed1_first5_match": True, "runif_seed42_first2_match": True}


def r_round2(x):
    """R's round(x, 2) — correct decimal rounding of the double, half-even.
    Python's round() implements the same correct rounding (David Gay)."""
    return round(x, 2)


# ===========================================================================
# Loading / generating the two datasets.
# ===========================================================================
def load_dmitriev():
    with open(DMITRIEV_CSV, "rb") as f:
        raw = f.read()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == DMITRIEV_SHA256, \
        f"data/dmitriev2019_psm_n52.csv has been modified (sha256 {sha})"
    rows = [r for r in csv.reader(raw.decode().splitlines())
            if r and not r[0].startswith("#") and r[0] != "respondent"]
    assert len(rows) == 52
    tc, ch, ex, te = (np.array([int(r[k]) for r in rows], float)
                      for k in (1, 2, 3, 4))
    # Certify the transcription: reproduce the article's Table 2 estimates
    # with the authors' own estimator, x_hat = (N+1)-th order statistic of
    # the pooled 2N-value sample of the two crossing columns (their eq. (8)).
    cols = {"X1": tc, "X2": ch, "X3": ex, "X4": te}
    pooled = {
        "x_min(X1,X2)": float(np.sort(np.r_[cols["X1"], cols["X2"]])[52]),
        "x_opt(X1,X4)": float(np.sort(np.r_[cols["X1"], cols["X4"]])[52]),
        "x_bezr(X3,X2)": float(np.sort(np.r_[cols["X3"], cols["X2"]])[52]),
        "x_max(X3,X4)": float(np.sort(np.r_[cols["X3"], cols["X4"]])[52]),
    }
    for k, v in DMITRIEV_TABLE2.items():
        assert pooled[k] == v, f"published {k}={v}, transcription gives {pooled[k]}"
    return tc, ch, ex, te, sha, pooled


def generate_vignette(save_path):
    """Bit-exact reproduction of the pricesensitivitymeter vignette draw."""
    rng = RRng(20)                                   # set.seed(20)
    tc = [r_round2(v) for v in rng.rnorm(250, 8, 1.5)]
    ch = [r_round2(v) for v in rng.rnorm(250, 12, 2)]
    ex = [r_round2(v) for v in rng.rnorm(250, 13, 1)]
    te = [r_round2(v) for v in rng.rnorm(250, 15, 1)]
    with open(save_path, "w", newline="") as f:
        f.write("# SIMULATED data — exact replication of the R draw in the\n"
                "# pricesensitivitymeter vignette 'Visualizing PSM Results'\n"
                "# (vignettes/visualizing-psm-results.Rmd, github.com/alletsee/"
                "pricesensitivitymeter,\n"
                f"# accessed {ACCESS_DATE}): set.seed(20); round(rnorm(250, m, s), 2)\n"
                "# with (m,s) = (8,1.5), (12,2), (13,1), (15,1). The package ships\n"
                "# NO real survey data; all its examples are simulated.\n"
                "# Generated by scripts/04_empirical_range.py (R RNG replicated in\n"
                "# Python and verified against known R outputs in-script).\n")
        w = csv.writer(f)
        w.writerow(["respondent", "toocheap", "cheap", "expensive", "tooexpensive"])
        for i in range(250):
            w.writerow([i + 1, tc[i], ch[i], ex[i], te[i]])
    return (np.array(tc), np.array(ch), np.array(ex), np.array(te))


def certify_vignette_replication(tc, ch, ex, te):
    """Certify the replicated draw against the package's PUBLISHED rendered
    vignette (max-alletsee.github.io/pricesensitivitymeter/articles/
    visualizing-psm-results.html, accessed 2026-07-13), which prints row 217
    of psm_analysis(validate=TRUE)$data_vanwestendorp:
        price 12.31, ecdf_toocheap 0, ecdf_cheap 0.2302158,
        ecdf_expensive 0.1510791, ecdf_tooexpensive 0.007194245
    (the printed 'ecdf' columns for toocheap/cheap are the descending
    survival shares). psm_analysis's validate rule is STRICT
    (tooexpensive > expensive > cheap > toocheap)."""
    strict = (tc < ch) & (ch < ex) & (ex < te)
    tcs, chs, exs, tes = tc[strict], ch[strict], ex[strict], te[strict]
    pooled = np.unique(np.concatenate([tcs, chs, exs, tes]))
    assert int(strict.sum()) == 139, f"strict-valid n = {strict.sum()} != 139"
    assert len(pooled) == 434, f"pooled unique prices {len(pooled)} != 434"
    p = float(pooled[216])                      # row 217, 1-indexed
    assert p == 12.31, f"row-217 price {p} != 12.31"
    checks = {
        "toocheap_surv": (float((tcs > p).mean()), 0.0),
        "cheap_surv": (float((chs > p).mean()), 0.2302158),
        "expensive_ecdf": (float((exs <= p).mean()), 0.1510791),
        "tooexpensive_ecdf": (float((tes <= p).mean()), 0.007194245),
    }
    for k, (got, want) in checks.items():
        assert abs(got - want) < 5e-8, f"{k}: {got} != published {want}"
    return {
        "published_vignette_html": "https://max-alletsee.github.io/"
            "pricesensitivitymeter/articles/visualizing-psm-results.html",
        "html_access_date": "2026-07-13",
        "strict_valid_n_matches_139": True,
        "pooled_unique_prices_matches_434": True,
        "row217_price_matches_12.31": True,
        "row217_curve_values_match_published": {k: g for k, (g, _) in checks.items()},
        "conclusion": "replicated draw is bit-exact vs the package's own "
                      "rendered vignette output",
    }


# ===========================================================================
# The paper's computations.
# ===========================================================================
def exact_optimum(w):
    """argmax_p p * P(W >= p) by EXACT atom enumeration (Fraction arithmetic).

    For discrete W the revenue p*P(W>=p) is piecewise-linear increasing
    between atoms, so the max over p > 0 is attained at an atom (draft.tex,
    proof of Theorem 1). Ties resolve to the smallest maximizer (paper
    convention). Returns (p_star, revenue, demand_at_p_star) as floats plus
    the full atom table.
    """
    ws = sorted(w.tolist())
    n = len(ws)
    atoms = sorted(set(ws))
    table = []
    best = None
    for a in atoms:
        # count of W >= a, exact
        k = n - np.searchsorted(np.asarray(ws), a, side="left")
        rev = Fraction(a).limit_denominator(10**9) * Fraction(int(k), n)
        table.append((a, int(k), rev))
        if best is None or rev > best[2]:
            best = (a, int(k), rev)
    p_star, k_star, rev_star = best
    return (float(p_star), float(rev_star), k_star / n,
            [(float(a), k, float(r)) for a, k, r in table])


def revenue_at(w, p):
    """R(p) = p * P(W >= p), recomputed two independent ways, asserted equal."""
    d1 = float((np.asarray(w) >= p).mean())
    d2 = (len(w) - int(np.searchsorted(np.sort(np.asarray(w)), p, side="left"))) / len(w)
    assert abs(d1 - d2) < 1e-12
    return p * d1, d1


def analyze(name, tc, ch, ex, te, grid_hi_pad=1.05, n_grid=40001):
    """Full pipeline for one dataset: ORD handling, curves, OPP, range."""
    n_raw = len(tc)
    ord_weak = (tc <= ch) & (ch <= ex) & (ex <= te)          # paper's ORD
    ord_strict = (tc < ch) & (ch < ex) & (ex < te)           # R package rule
    n_weak_viol = int(n_raw - ord_weak.sum())
    n_strict_viol = int(n_raw - ord_strict.sum())
    keep = ord_weak                                          # drop ORD violators
    tc, ch, ex, te = tc[keep], ch[keep], ex[keep], te[keep]
    n = len(tc)

    hi = float(max(tc.max(), ch.max(), ex.max(), te.max())) * grid_hi_pad
    grid = np.linspace(0.0, hi, n_grid)
    step = grid[1] - grid[0]

    psm = psm_points(tc, ch, ex, te, grid=grid)

    # tc/te marginal overlap diagnostic (paper Remark 3: the OPP is a
    # positive-height crossing iff the too-cheap and too-expensive marginals
    # overlap across respondents).
    overlap = {
        "max_tc": float(tc.max()), "min_te": float(te.min()),
        "marginals_overlap": bool(tc.max() >= te.min()),
        "share_tc_at_or_above_min_te": float((tc >= te.min()).mean()),
    }

    # --- verification: curve monotonicity, bounds, cone-compatibility ------
    for c in ("toocheap", "cheap"):
        assert np.all(np.diff(psm[c]) <= 1e-15), f"{c} not non-increasing"
    for c in ("expensive", "tooexpensive"):
        assert np.all(np.diff(psm[c]) >= -1e-15), f"{c} not non-decreasing"
    for c in ("toocheap", "cheap", "expensive", "tooexpensive"):
        assert psm[c].min() >= 0 and psm[c].max() <= 1
    F_tc, F_ch = 1 - psm["toocheap"], 1 - psm["cheap"]
    F_ex, F_te = psm["expensive"], psm["tooexpensive"]
    assert np.all(F_tc >= F_ch - 1e-12) and np.all(F_ch >= F_ex - 1e-12) \
        and np.all(F_ex >= F_te - 1e-12), "marginals not cone-compatible"

    OPP, OPP_h = psm["OPP"]
    # original-convention acceptable range (S_tc vs F_ch; F_te vs S_ex),
    # cf. draft.tex Section 2.2 — reported for completeness.
    pmc_orig = _first_crossing(grid, psm["toocheap"], 1 - psm["cheap"])
    pme_orig = _first_crossing(grid, psm["tooexpensive"], 1 - psm["expensive"])

    # --- the identified range: one revenue optimum per declared reading ----
    readings = {}
    for label, w in (("ch", ch), ("ex", ex), ("te", te)):
        p_star, rev_star, d_star, _table = exact_optimum(w)
        # cross-check the exact atom enumeration against the grid maximizer
        Dg = np.array([(w >= p).mean() for p in grid[:: max(1, n_grid // 4001)]])
        gg = grid[:: max(1, n_grid // 4001)]
        i = int(np.argmax(gg * Dg))
        assert gg[i] * Dg[i] <= rev_star + 1e-9, "grid beats exact atom max?"
        r_chk, _ = revenue_at(w, p_star)
        assert abs(r_chk - rev_star) < 1e-9
        r_at_opp, d_at_opp = revenue_at(w, OPP)
        readings[label] = {
            "p_star": p_star, "revenue_at_p_star": rev_star,
            "demand_at_p_star": d_star,
            "revenue_at_OPP": r_at_opp, "demand_at_OPP": d_at_opp,
            "loss_pricing_at_OPP": 1 - r_at_opp / rev_star,
        }

    p_ch, p_te = readings["ch"]["p_star"], readings["te"]["p_star"]
    lo, hi_ = min(p_ch, p_te), max(p_ch, p_te)
    out = {
        "n_raw": n_raw, "n_after_ORD_drop": n,
        "ORD_violations_weak_paper_rule_dropped": n_weak_viol,
        "ORD_violations_strict_R_package_rule_reported_only": n_strict_viol,
        "grid": {"lo": 0.0, "hi": float(grid[-1]), "n": n_grid,
                 "step": float(step),
                 "note": "curves/crossings on this grid; revenue optima by "
                         "exact atom enumeration over the observed support, "
                         "grid used only as a cross-check"},
        "tc_te_overlap_remark3": overlap,
        "crossing_convention_note": "step curves can satisfy the crossing "
            "equation on a whole interval; psm_core (like the R package's "
            "intersection_method='min') reports the LEFT end, the Dmitriev "
            "et al. estimator (their eq. 8) the RIGHT end. Both are reported "
            "for the real dataset; every loss figure is computed at both.",
        "psm_points": {k: {"price": psm[k][0], "height": psm[k][1]}
                       for k in ("PMC", "OPP", "IPP", "PME")},
        "psm_range_original_convention": {
            "PMC_orig": {"price": pmc_orig[0], "height": pmc_orig[1]} if pmc_orig else None,
            "PME_orig": {"price": pme_orig[0], "height": pme_orig[1]} if pme_orig else None,
        },
        "readings": readings,
        "identified_range": {
            "p_star_ch": p_ch, "p_star_te": p_te,
            "width_abs": hi_ - lo, "ratio_te_over_ch": p_te / p_ch,
            "OPP": OPP, "OPP_height": OPP_h,
            "OPP_inside_range": bool(lo <= OPP <= hi_),
            "OPP_position_in_range_0is_pch_1is_pte":
                (OPP - p_ch) / (p_te - p_ch) if p_te != p_ch else None,
        },
    }
    return out, psm, grid, (tc, ch, ex, te)


def main():
    print("=" * 72)
    print("EMPIRICAL IDENTIFIED-RANGE COMPUTATION (Section 8 recommendation)")
    print("=" * 72)

    log = {"access_date": ACCESS_DATE,
           "r_rng_verification": verify_r_rng()}

    # ---------------- dataset 1: real survey data --------------------------
    tc, ch, ex, te, sha, pooled = load_dmitriev()
    res1, psm1, grid1, thr1 = analyze("dmitriev2019", tc, ch, ex, te)
    res1_src = {
        "dataset": "dmitriev2019_psm_n52",
        "real_or_simulated": "REAL survey data (published rescaled: prices in "
            "conventional units, common multiplicative rescaling for "
            "commercial confidentiality; all ratios/percentages invariant)",
        "description": "PSM survey of representatives of 52 organizations on "
            "a new B2B software product after a two-month trial; full "
            "respondent-level Table 1 of the article",
        "citation": "Yu.G. Dmitriev, Zh.N. Zenkova, A.G. Zenkov (2019), "
            "Statistical properties of price sensitivity meter results, "
            "Tomsk State University Journal of Control and Computer Science "
            "47, 30-40",
        "doi": "10.17223/19988605/47/4",
        "source_url": "https://cyberleninka.ru/article/n/statisticheskie-"
            "svoystva-otsenok-metoda-izmereniya-tsenovoy-chuvstvitelnosti-psm/pdf",
        "access_date": ACCESS_DATE,
        "file": "data/dmitriev2019_psm_n52.csv", "sha256": sha,
        "column_pairing_caveat": "published table lists each column sorted "
            "ascending; respondent pairing not recoverable; every reported "
            "quantity is a functional of the marginals alone (paper Lemma 1) "
            "and is unaffected; under the as-printed (comonotone) pairing "
            "0/52 rows violate ORD",
        "transcription_certificate": {
            "method": "authors' estimator, (N+1)-th order statistic of the "
                      "pooled 2N sample (their eq. 8)",
            "reproduced": pooled, "published_table2": DMITRIEV_TABLE2,
            "all_match": True},
    }
    log["dmitriev2019"] = {"source": res1_src, **res1}

    print("\n[1] REAL DATA — Dmitriev, Zenkova & Zenkov (2019), N = 52")
    print(f"    transcription certified against published Table 2: {pooled}")
    print(f"    ORD violations dropped (weak, paper rule): "
          f"{res1['ORD_violations_weak_paper_rule_dropped']}")
    for k in ("PMC", "OPP", "IPP", "PME"):
        print(f"    {k}: {fmt_pt((res1['psm_points'][k]['price'], res1['psm_points'][k]['height']))}")
    ir = res1["identified_range"]
    print(f"    p*_ch = {ir['p_star_ch']:.4f}   p*_te = {ir['p_star_te']:.4f}   "
          f"(p*_ex = {res1['readings']['ex']['p_star']:.4f})")
    print(f"    identified range width = {ir['width_abs']:.4f}, "
          f"ratio p*_te/p*_ch = {ir['ratio_te_over_ch']:.4f}")
    print(f"    OPP = {ir['OPP']:.4f} (height {ir['OPP_height']:.4f}), "
          f"inside range: {ir['OPP_inside_range']}, position "
          f"{ir['OPP_position_in_range_0is_pch_1is_pte']:.3f}")
    for lab in ("ch", "ex", "te"):
        r = res1["readings"][lab]
        print(f"    loss pricing at OPP under W={lab}: "
              f"{100 * r['loss_pricing_at_OPP']:.1f}%  "
              f"(R(OPP)={r['revenue_at_OPP']:.2f} vs R*={r['revenue_at_p_star']:.2f})")

    # losses at the price point the STUDY ACTUALLY REPORTED (their x_opt=290)
    published_opp = float(DMITRIEV_TABLE2["x_opt(X1,X4)"])
    tc1, ch1, ex1, te1 = thr1
    pub = {}
    for lab, w in (("ch", ch1), ("ex", ex1), ("te", te1)):
        r_at, d_at = revenue_at(w, published_opp)
        r_star = log["dmitriev2019"]["readings"][lab]["revenue_at_p_star"]
        pub[lab] = {"revenue_at_published_OPP": r_at,
                    "demand_at_published_OPP": d_at,
                    "loss_pricing_at_published_OPP": 1 - r_at / r_star}
        print(f"    loss pricing at PUBLISHED x_opt=290 under W={lab}: "
              f"{100 * pub[lab]['loss_pricing_at_published_OPP']:.1f}%")
    log["dmitriev2019"]["published_OPP_290"] = {
        "value": published_opp,
        "note": "the article's own reported optimal price (Table 2, x_opt), "
                "computed with their pooled-order-statistic estimator",
        "losses_by_reading": pub}

    # interval-attainment certificate: every price q in [p*_ch, p*_te] is the
    # EXACT optimum of some coherent reading (take W_i = clamp(q, tc_i, te_i);
    # then R(q) = q and R(p) < q for all p in the grid, p != q).
    lo_att = log["dmitriev2019"]["identified_range"]["p_star_ch"]
    hi_att = log["dmitriev2019"]["identified_range"]["p_star_te"]
    qs = [lo_att + (hi_att - lo_att) * k / 4000 for k in range(4001)]
    for q in qs:
        tc_a = np.array(tc1, dtype=float)
        te_a = np.array(te1, dtype=float)
        in_window = (tc_a <= q) & (q <= te_a)
        w_read = np.where(in_window, q, tc_a)
        p_star_q = exact_optimum(w_read)[0]
        assert abs(p_star_q - q) < 1e-9, (q, p_star_q)
    log["dmitriev2019"]["interval_attainment_certificate"] = {
        "certified": True, "n_grid": len(qs),
        "range": [lo_att, hi_att],
        "construction": "W_i = q when tc_i <= q <= te_i, else W_i = tc_i; "
            "coherent since tc_i <= W_i <= te_i either way. The exact-atom "
            "optimizer returns q for every q on a 4001-point grid over the "
            "identified range, so every price in the range is the exact "
            "optimum of some coherent reading."}
    print(f"    interval-attainment certificate: all {len(qs)} grid prices "
          f"in [{lo_att:.0f}, {hi_att:.0f}] attained by coherent readings")

    # ---------------- dataset 2: package vignette (SIMULATED) --------------
    vign_csv = os.path.join(DATA, "psm_package_vignette_sim_n250.csv")
    vtc, vch, vex, vte = generate_vignette(vign_csv)
    vign_cert = certify_vignette_replication(vtc, vch, vex, vte)
    res2, psm2, grid2, thr2 = analyze("vignette_sim", vtc, vch, vex, vte,
                                      n_grid=40001)
    log["pricesensitivitymeter_vignette"] = {
        "source": {
            "dataset": "psm_package_vignette_sim_n250",
            "real_or_simulated": "SIMULATED (the R package ships no real "
                "survey data; every documentation example is simulated). "
                "This is the package's canonical vignette example, "
                "reproduced bit-exactly by replicating R's RNG.",
            "citation": "M. Alletsee, pricesensitivitymeter: Van Westendorp "
                "Price Sensitivity Meter Analysis, R package version 1.3.3; "
                "vignette 'Visualizing PSM Results'",
            "source_url": "https://raw.githubusercontent.com/alletsee/"
                "pricesensitivitymeter/master/vignettes/"
                "visualizing-psm-results.Rmd",
            "access_date": ACCESS_DATE,
            "simulation": "set.seed(20); round(rnorm(250, m, s), 2), (m,s) = "
                "toocheap (8,1.5), cheap (12,2), expensive (13,1), "
                "tooexpensive (15,1); independent draws",
            "file": "data/psm_package_vignette_sim_n250.csv",
            "sha256": hashlib.sha256(open(vign_csv, "rb").read()).hexdigest(),
            "replication_certificate": vign_cert,
        },
        **res2}

    print(f"\n[2] SIMULATED DATA — pricesensitivitymeter vignette, n = 250 raw")
    print(f"    replication certified vs published rendered vignette "
          f"(n=139 strict-valid, 434 pooled prices, row 217 = 12.31, "
          f"all four curve values match): "
          f"{vign_cert['conclusion']}")
    print(f"    ORD violations dropped (weak, paper rule): "
          f"{res2['ORD_violations_weak_paper_rule_dropped']} "
          f"-> n = {res2['n_after_ORD_drop']}  (strict R-package rule would "
          f"drop {res2['ORD_violations_strict_R_package_rule_reported_only']})")
    for k in ("PMC", "OPP", "IPP", "PME"):
        print(f"    {k}: {fmt_pt((res2['psm_points'][k]['price'], res2['psm_points'][k]['height']))}")
    ir2 = res2["identified_range"]
    print(f"    p*_ch = {ir2['p_star_ch']:.4f}   p*_te = {ir2['p_star_te']:.4f}   "
          f"(p*_ex = {res2['readings']['ex']['p_star']:.4f})")
    print(f"    identified range width = {ir2['width_abs']:.4f}, "
          f"ratio p*_te/p*_ch = {ir2['ratio_te_over_ch']:.4f}")
    print(f"    OPP = {ir2['OPP']:.4f} (height {ir2['OPP_height']:.4f}), "
          f"inside range: {ir2['OPP_inside_range']}, position "
          f"{ir2['OPP_position_in_range_0is_pch_1is_pte']:.3f}")
    for lab in ("ch", "ex", "te"):
        r = res2["readings"][lab]
        print(f"    loss pricing at OPP under W={lab}: "
              f"{100 * r['loss_pricing_at_OPP']:.1f}%")

    # ---------------- persist ----------------------------------------------
    out_path = os.path.join(RESULTS, "04_empirical_range.json")
    with open(out_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nwrote results/04_empirical_range.json")
    print(f"wrote data/psm_package_vignette_sim_n250.csv (regenerated, deterministic)")

    # ---------------- optional figure (real dataset; black and white) -------
    # Changed 2026-11-07 for the JRPM version of the paper: greyscale with
    # distinct line styles, larger fonts, 6.5 x 2.8 in, vector PDF/EPS plus a
    # 600 dpi TIFF without alpha (journal rule: black and white only). The
    # right panel is titled 'Focal range': the spread across the two declared
    # readings, not the set of prices the data permit (script 05 computes that
    # set). The numerical outputs above are unchanged.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update(_RC)
        tc1, ch1, ex1, te1 = thr1
        fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.8))
        g = grid1
        for key, lab, col, ls in _CURVE_STYLES:
            ax[0].plot(g, psm1[key], color=col, ls=ls, lw=1.1, label=lab)
        ax[0].axvline(published_opp, color="black", ls="--", lw=0.8)
        ax[0].text(published_opp + 8, 1.03, f"reported OPP {published_opp:.0f}",
                   fontsize=6.5, ha="left", va="bottom")
        ax[0].set_title("Four PSM curves, real survey data (N = 52)")
        ax[0].set_xlabel("price (conventional units)")
        ax[0].set_ylabel("share")
        ax[0].legend(loc="center right", bbox_to_anchor=(1.0, 0.45), frameon=False)
        ax[0].set_xlim(0, 700); ax[0].set_ylim(0, 1.25)
        ax[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])

        for lab, w, col, ls in (("ch", ch1, "black", "-"), ("te", te1, "0.45", "--")):
            D = np.array([(w >= p).mean() for p in g])
            ax[1].plot(g, g * D, color=col, ls=ls, lw=1.1, label=f"W = {lab}")
            ax[1].axvline(res1["readings"][lab]["p_star"], color=col, ls=":", lw=1.2)
        ax[1].axvline(published_opp, color="black", ls="--", lw=0.8,
                      label=f"reported OPP {published_opp:.0f}")
        ax[1].set_title("Focal range: one optimum per reading")
        ax[1].set_xlabel("price (conventional units)")
        ax[1].set_ylabel("p * D(p)")
        ax[1].legend(loc="lower right", frameon=False)
        ax[1].set_xlim(0, 700)
        fig.tight_layout()
        _save_figure(fig, os.path.join(RESULTS, "04_empirical_range"))
        print("wrote results/04_empirical_range.pdf, .eps, .png, .tif")
    except Exception as e:  # noqa
        print("figure skipped:", e)

    print("\nHEADLINE (real data): identified range "
          f"[{ir['p_star_ch']:.0f}, {ir['p_star_te']:.0f}] "
          f"(ratio {ir['ratio_te_over_ch']:.2f}); OPP {ir['OPP']:.0f} costs "
          f"{100 * res1['readings']['ch']['loss_pricing_at_OPP']:.1f}% of revenue "
          f"under the W=ch reading and "
          f"{100 * res1['readings']['te']['loss_pricing_at_OPP']:.1f}% under W=te.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
