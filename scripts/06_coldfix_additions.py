"""
06_coldfix_additions.py - exact-arithmetic results added in the fix round of
2026-11-13 (cold referee report). Pure Python (fractions, csv, json, math,
random); no numpy.

What it computes:

  1. The band-wide minimax relative-regret price (Proposition 2). For a band
     lo <= W <= hi write S_lo(p) = Pr(lo >= p), S_hi(p) = Pr(hi >= p),
     g(p) = p S_lo(p), h(q) = q S_hi(q). The worst-case share of attainable
     revenue forgone at price p over every law in the band is
         rho(p) = 1 - min( inf_{q<p} g(p)/h(q),
                           inf_{q>p} max(p/q, g(p)/h(q)), 1 ).
     Computed at every answer value (the maximum of the guaranteed share
     1 - rho is attained at an answer value), for the published survey under
     the bands [tc, te] and [ch, ex] and for the Table 1 population under
     [tc, te]. Each value is certified two ways: an explicit demand function
     in the band attains (or approaches) it, and 3000 random coherent
     readings never exceed it.
  2. The crossing convention OPP = inf{p : S_tc(p) - F_te(p) < 0}, the lower
     end inf{p : difference <= 0}, and the crossing height
     min{S_tc(OPP), F_te(OPP)} as a respondent count, for all four price
     points of the survey and of Table 1.
  3. Ties and ordering under the as-printed pairing of the survey; the
     readings' optima when tied rows are dropped.
  4. The alternative demand D(p) = Pr(tc < p <= W) (no purchase below the
     too-cheap price): optima of the readings W = ch and W = te for Table 1
     and the survey.
  5. Population values of Section 7 (no sampling): OPP, the two optima and
     the two losses for the normal-mixture example.
  6. The two-segment family of Proposition 3 (loss at the OPP under a
     declared reading tends to 1).
  7. The three-reading minimax price 257: why it sits at the smallest
     walk-away answer.

Input:  data/dmitriev2019_psm_n52.csv (sha256 checked).
Output: results/06_coldfix_additions.json
Run:    python3 scripts/06_coldfix_additions.py
"""
import csv
import hashlib
import json
import math
import os
import random
import sys
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
RESULTS = os.path.join(HERE, "..", "results")
CSV = os.path.join(DATA, "dmitriev2019_psm_n52.csv")
CSV_SHA256 = "45c4edca85aecd669d816bc3e81c0e5f91edf4a89cf9564316d8c13fddd96337"


def surv(xs, p):          # Pr(X >= p)
    return Fr(sum(1 for x in xs if x >= p), len(xs))


def cdf(xs, p):           # Pr(X <= p)
    return Fr(sum(1 for x in xs if x <= p), len(xs))


def revenue(xs, p):
    return p * surv(xs, p)


def optimum(xs):
    best = None
    for a in sorted(set(xs)):
        r = revenue(xs, a)
        if best is None or r > best[1]:
            best = (a, r)
    return best


# ---------------------------------------------------------------- 1. minimax
def guaranteed_share(lo, hi, p):
    """1 - rho(p): the infimum over the band of R_F(p) / sup_x R_F(x).
    Returns (share, worst q, kind)."""
    atoms_hi = sorted(set(hi))
    s = surv(lo, p)
    g = p * s
    best = (Fr(1), p, "self")
    # q < p: ratio g(p)/h(q); sup of h on q < p is at answers below p or the
    # left limit q -> p-, where h -> p S_hi(p)
    for q in atoms_hi:
        if q < p and revenue(hi, q) > 0:
            r = g / revenue(hi, q)
            if r < best[0]:
                best = (r, q, "below")
    if surv(hi, p) > 0:
        r = s / surv(hi, p)
        if r < best[0]:
            best = (r, p, "left-limit")
    # q > p: ratio max(p/q, g(p)/h(q))
    for q in atoms_hi:
        if q > p and surv(hi, q) > 0:
            r = max(Fr(p) / q, g / revenue(hi, q))
            if r < best[0]:
                best = (r, q, "above")
    return best


def adversary_regret(lo, hi, p, q, kind):
    """Exact share forgone at p under the explicit band member built for
    (p, q). Demand D is evaluated at every answer value and at p."""
    pts = sorted(set(lo) | set(hi) | {p, q})
    def D(x):
        if kind == "below":            # q < p
            return surv(hi, x) if x <= q else surv(lo, x)
        if kind == "above":            # q > p
            if x <= p or x > q:
                return surv(lo, x)
            return max(surv(lo, x), min(surv(hi, q), surv(lo, p)))
        raise ValueError
    vals = [D(x) for x in pts]
    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
    assert all(surv(lo, x) <= D(x) <= surv(hi, x) for x in pts)
    rmax = max(x * D(x) for x in pts)
    return 1 - p * D(p) / rmax


def minimax_block(lo, hi, label, rng):
    cands = sorted(set(lo) | set(hi))
    table = {}
    for p in cands:
        share, q, kind = guaranteed_share(lo, hi, p)
        table[p] = (share, q, kind)
    p_hat = max(cands, key=lambda p: (table[p][0], -p))
    share, q, kind = table[p_hat]
    out = {"band": label, "minimax_price": str(p_hat),
           "guaranteed_share": str(share), "guaranteed_share_float": float(share),
           "worst_case_share_forgone_pct": round(float(1 - share) * 100, 2),
           "worst_case_comparison_price": str(q), "worst_case_kind": kind}
    # certificate 1: the explicit adversary attains rho at every candidate
    # where the worst case is an attained one (kinds below/above)
    n_att = 0
    for p in cands:
        sh, qq, kd = table[p]
        if kd in ("below", "above"):
            assert adversary_regret(lo, hi, p, qq, kd) == 1 - sh, (label, p)
            n_att += 1
    out["explicit_adversary_attains_rho_at_n_candidates"] = n_att
    out["candidates"] = len(cands)
    # certificate 2: random coherent readings never exceed rho
    slo, shi = sorted(lo), sorted(hi)
    assert all(a <= b for a, b in zip(slo, shi))
    worst_seen = {p: Fr(0) for p in cands}
    for t in range(3000):
        mode = t % 4
        if mode == 0:
            W = [rng.randint(a, b) for a, b in zip(slo, shi)]
        elif mode == 1:
            W = [a if rng.random() < 0.5 else b for a, b in zip(slo, shi)]
        elif mode == 2:
            c = rng.choice(cands)
            W = [min(max(c, a), b) for a, b in zip(slo, shi)]
        else:
            k = rng.randrange(len(slo) + 1)
            W = list(slo[:k]) + list(shi[k:])
            if rng.random() < 0.5:
                W = list(shi[:k]) + [max(a, min(b, shi[k - 1] if k else a)) for a, b in zip(slo[k:], shi[k:])]
        assert all(a <= w <= b for a, w, b in zip(slo, W, shi))
        rstar = optimum(W)[1]
        for p in cands:
            reg = 1 - revenue(W, p) / rstar
            if reg > worst_seen[p]:
                worst_seen[p] = reg
    assert all(worst_seen[p] <= 1 - table[p][0] for p in cands)
    out["random_readings_checked"] = 3000
    out["random_search_largest_regret_at_minimax_price_pct"] = round(float(worst_seen[p_hat]) * 100, 2)
    # a few rows for the text
    out["rows"] = {str(p): {"guaranteed_share_pct": round(float(table[p][0]) * 100, 2),
                            "worst_case_share_forgone_pct": round(float(1 - table[p][0]) * 100, 2),
                            "worst_q": str(table[p][1]), "kind": table[p][2]}
                   for p in cands}
    out["max_g"] = str(max(revenue(lo, a) for a in set(lo)))
    out["max_h"] = str(max(revenue(hi, a) for a in set(hi)))
    return out, table


# ------------------------------------------------------------- 2. crossings
def crossing(desc, asc):
    """desc: answers of the descending curve S(p) = Pr(X >= p); asc: answers
    of the ascending curve F(p) = Pr(Y <= p). d(p) = S(p) - F(p) is
    nonincreasing. Returns (p_lo, p_hi, height count)."""
    n = len(desc)
    pts = sorted(set(desc) | set(asc))
    def d(p):
        return surv(desc, p) - cdf(asc, p)
    # d is piecewise constant: constant on open gaps between consecutive
    # points; evaluate at points and at gap midpoints
    probes = []
    for i, x in enumerate(pts):
        probes.append(Fr(x))
        if i + 1 < len(pts):
            probes.append((Fr(x) + pts[i + 1]) / 2)
    probes = [Fr(pts[0]) - 1] + probes + [Fr(pts[-1]) + 1]
    neg = [x for x in probes if d(x) < 0]
    nonpos = [x for x in probes if d(x) <= 0]
    def inf_of(first):
        # the infimum is the answer value at or just below the first probe
        if first in [Fr(x) for x in pts]:
            return first
        return max(Fr(x) for x in pts if x < first)
    p_hi = inf_of(min(neg))
    p_lo = inf_of(min(nonpos))
    height = min(surv(desc, p_hi), cdf(asc, p_hi))
    return p_lo, p_hi, height, int(height * n) if (height * n).denominator == 1 else float(height * n)


# -------------------------------------------------------- 5. population values
def Phi(z):
    return 0.5 * math.erfc(-z / math.sqrt(2))


def mix_sf(x):            # Pr(v >= x), v ~ 0.6 N(10, 2^2) + 0.4 N(24, 3^2)
    return 0.6 * (1 - Phi((x - 10) / 2)) + 0.4 * (1 - Phi((x - 24) / 3))


def argmax_1d(f, a, b, n=200001):
    best_x, best = a, f(a)
    for i in range(1, n):
        x = a + (b - a) * i / (n - 1)
        v = f(x)
        if v > best:
            best_x, best = x, v
    # refine
    step = (b - a) / (n - 1)
    lo_, hi_ = best_x - step, best_x + step
    for _ in range(200):
        m1 = lo_ + (hi_ - lo_) / 3
        m2 = hi_ - (hi_ - lo_) / 3
        if f(m1) < f(m2):
            lo_ = m1
        else:
            hi_ = m2
    x = (lo_ + hi_) / 2
    return x, f(x)


def main():
    raw = open(CSV, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == CSV_SHA256
    rows = [r for r in csv.reader(raw.decode().splitlines())
            if r and not r[0].startswith("#") and r[0] != "respondent"]
    assert len(rows) == 52
    tc = [int(r[1]) for r in rows]; ch = [int(r[2]) for r in rows]
    ex = [int(r[3]) for r in rows]; te = [int(r[4]) for r in rows]
    wtc = [4] * 7 + [8] * 5 + [12] * 5 + [16] * 3
    wch = [7] * 7 + [12] * 5 + [18] * 5 + [24] * 3
    wex = [10] * 7 + [16] * 5 + [24] * 5 + [32] * 3
    wte = [14] * 7 + [22] * 5 + [32] * 5 + [44] * 3
    rng = random.Random(20261113)
    log = {"data_file": "data/dmitriev2019_psm_n52.csv", "sha256": sha, "n": 52}

    # 1
    mm = {}
    for label, lo, hi in (("survey_COH_tc_te", tc, te), ("survey_inner_ch_ex", ch, ex),
                          ("witness_COH_tc_te", wtc, wte), ("witness_inner_ch_ex", wch, wex)):
        mm[label], tab = minimax_block(lo, hi, label, rng)
        def row(p, lo=lo, hi=hi):
            sh, q, kd = guaranteed_share(lo, hi, p)
            return {"guaranteed_share_pct": round(float(sh) * 100, 2),
                    "worst_case_share_forgone_pct": round(float(1 - sh) * 100, 2),
                    "worst_q": str(q), "kind": kd}
        if label.startswith("survey"):
            mm[label]["text_rows"] = {str(p): row(p) for p in (125, 251, 253, 257, 290, 301)}
        else:
            mm[label]["text_rows"] = {str(p): row(p) for p in (8, 12, 14, 22)}
    log["band_wide_minimax_relative_regret"] = mm

    # 2
    cr = {}
    for label, (a, b, c, d) in (("survey", (tc, ch, ex, te)), ("witness", (wtc, wch, wex, wte))):
        block = {}
        for name, desc, asc in (("OPP", a, d), ("IPP", b, c), ("PMC", a, c), ("PME", b, d)):
            p_lo, p_hi, h, k = crossing(desc, asc)
            block[name] = {"inf_diff_nonpositive": str(p_lo), "inf_diff_negative": str(p_hi),
                           "height": str(h), "height_in_respondents_or_weight_x_n": k}
        cr[label] = block
    assert cr["witness"]["OPP"]["inf_diff_negative"] == "14" and Fr(cr["witness"]["OPP"]["height"]) == Fr(3, 20)
    assert [cr["witness"][k]["inf_diff_negative"] for k in ("PMC", "OPP", "IPP", "PME")] == ["12", "14", "16", "18"]
    log["crossing_convention"] = cr
    log["overlap"] = {"survey_max_tc": max(tc), "survey_min_te": min(te),
                      "survey_tc_answers_at_or_above_min_te": sum(1 for x in tc if x >= min(te)),
                      "witness_max_tc": 16, "witness_min_te": 14}

    # 3
    ties = [i + 1 for i, r in enumerate(zip(tc, ch, ex, te)) if not (r[0] < r[1] < r[2] < r[3])]
    keep = [i for i in range(52) if (i + 1) not in ties]
    block = {"rows_violating_ORD_as_printed": sum(1 for r in zip(tc, ch, ex, te) if not (r[0] <= r[1] <= r[2] <= r[3])),
             "rows_with_a_tie_as_printed": ties, "n_after_dropping_ties": len(keep)}
    if ties:
        for name, xs in (("ch", ch), ("ex", ex), ("te", te)):
            sub = [xs[i] for i in keep]
            p, r = optimum(sub)
            block[f"p_star_{name}_ties_dropped"] = str(p)
        stc = [tc[i] for i in keep]; ste = [te[i] for i in keep]
        p_lo, p_hi, h, k = crossing(stc, ste)
        block["OPP_ties_dropped"] = [str(p_lo), str(p_hi), str(h)]
    log["ties_and_ordering"] = block

    # 4
    def alt_opt(tcs, ws):
        pts = sorted(set(tcs) | set(ws))
        best = None
        for p in pts:
            dem = surv(ws, p) - surv(tcs, p)      # Pr(tc < p <= W), tc <= W
            r = p * dem
            if best is None or r > best[1]:
                best = (p, r)
        return best
    alt = {}
    for label, (a, b, c, d) in (("witness", (wtc, wch, wex, wte)), ("survey", (tc, ch, ex, te))):
        pa, ra = alt_opt(a, b); pb, rb = alt_opt(a, d)
        alt[label] = {"p_star_W_eq_ch": str(pa), "revenue_ch": str(ra),
                      "p_star_W_eq_te": str(pb), "revenue_te": str(rb)}
    def alt_loss(tcs, ws, p, rstar):
        return 1 - p * (surv(ws, p) - surv(tcs, p)) / rstar
    alt["witness"]["loss_at_OPP_14_ch_pct"] = round(float(alt_loss(wtc, wch, 14, Fr(alt["witness"]["revenue_ch"]))) * 100, 2)
    alt["witness"]["loss_at_OPP_14_te_pct"] = round(float(alt_loss(wtc, wte, 14, Fr(alt["witness"]["revenue_te"]))) * 100, 2)
    alt["survey"]["loss_at_290_ch_pct"] = round(float(alt_loss(tc, ch, 290, Fr(alt["survey"]["revenue_ch"]))) * 100, 2)
    alt["survey"]["loss_at_290_te_pct"] = round(float(alt_loss(tc, te, 290, Fr(alt["survey"]["revenue_te"]))) * 100, 2)
    # a coherent reading of Table 1 whose optimum under the alternative
    # demand lies below the OPP: W = 13 for T1-T3, W = 16 for T4
    W13 = [13] * 17 + [16] * 3
    assert all(a <= w <= b for a, w, b in zip(wtc, W13, wte))
    def alt_rev(tcs, ws, p):
        return p * Fr(sum(1 for t, w in zip(tcs, ws) if t < p <= w), len(ws))
    pts = sorted(set(wtc) | set(W13))
    best = max(pts, key=lambda p: (alt_rev(wtc, W13, p), -p))
    assert best == 13 and alt_rev(wtc, W13, 13) == Fr(221, 20)
    # no price off the answer values does better (fine rational grid)
    assert all(alt_rev(wtc, W13, Fr(k, 50)) <= Fr(221, 20) for k in range(1, 2500))
    alt["witness"]["reading_with_optimum_below_OPP"] = {
        "W_by_segment": [13, 13, 13, 16], "optimum": "13", "revenue": "221/20"}
    log["alternative_demand_tc_lt_p_le_W"] = alt

    # 5
    # tc = v-4, ch = v-1, ex = v+3, te = v+8; W_A = ch = v-1; W_B = te-1 = v+7
    def dOPP(p):          # Pr(tc >= p) - Pr(te <= p)
        return mix_sf(p + 4) - (1 - mix_sf(p - 8))
    a, b = 0.0, 40.0
    for _ in range(200):
        m = (a + b) / 2
        if dOPP(m) > 0:
            a = m
        else:
            b = m
    opp = (a + b) / 2
    RA = lambda p: p * mix_sf(p + 1)
    RB = lambda p: p * mix_sf(p - 7)
    pA, rA = argmax_1d(RA, 1.0, 45.0)
    pB, rB = argmax_1d(RB, 1.0, 45.0)
    log["section7_population_values"] = {
        "OPP": opp, "height_at_OPP": mix_sf(opp + 4),
        "p_star_A": pA, "R_star_A": rA, "p_star_B": pB, "R_star_B": rB,
        "loss_at_OPP_A_pct": round((1 - RA(opp) / rA) * 100, 2),
        "loss_at_OPP_B_pct": round((1 - RB(opp) / rB) * 100, 2),
        "second_local_max_A": argmax_1d(RA, 1.0, 14.0), "second_local_max_B": argmax_1d(RB, 20.0, 45.0)}

    # 6  two segments: weight 1-delta at (1,2,3,4), weight delta at (K, K+1, K+2, K+3)
    fam = []
    for delta, K in ((Fr(1, 5), 17), (Fr(1, 5), 97), (Fr(1, 5), 997), (Fr(1, 10), 37)):
        n = delta.denominator; k = delta.numerator
        a_ = [1] * (n - k) + [K] * k; d_ = [4] * (n - k) + [K + 3] * k
        p_lo, p_hi, h, _ = crossing(a_, d_)
        pst, rst = optimum(d_)
        loss = 1 - revenue(d_, p_hi) / rst
        assert p_hi == 4 and h == delta and loss == 1 - 4 / (delta * (K + 3))
        fam.append({"delta": str(delta), "K": K, "OPP": str(p_hi), "height": str(h),
                    "p_star_W_eq_te": str(pst), "loss_at_OPP": str(loss), "loss_pct": round(float(loss) * 100, 2)})
    log["proposition3_family"] = fam

    # 7  the three-reading minimax price
    Rs = {k: optimum(v)[1] for k, v in (("ch", ch), ("ex", ex), ("te", te))}
    def worst3(p):
        return max(1 - revenue(v, p) / Rs[k] for k, v in (("ch", ch), ("ex", ex), ("te", te)))
    log["three_reading_minimax"] = {
        "price": 257, "worst_pct": round(float(worst3(257)) * 100, 2),
        "smallest_walk_away_answer": min(te),
        "walk_away_loss_at_257": str(1 - revenue(te, 257) / Rs["te"]),
        "walk_away_demand_at_257": str(surv(te, 257)),
        "note": "for p <= 257 every respondent buys under W = te, so the walk-away loss is 1 - p/R*_te, decreasing in p; just above 257 the walk-away demand drops to 51/52 and the bargain loss keeps rising; the minimum of the maximum sits at the kink",
        "worst_at_256_pct": round(float(worst3(256)) * 100, 2),
        "worst_at_258_pct": round(float(worst3(258)) * 100, 2),
        "losses_at_257": {k: str(1 - revenue(v, 257) / Rs[k]) for k, v in (("ch", ch), ("ex", ex), ("te", te))}}

    # losses at the inner-band minimax price 236 under the three readings
    log["losses_at_236"] = {k: round(float(1 - revenue(v, 236) / Rs[k]) * 100, 2)
                            for k, v in (("ch", ch), ("ex", ex), ("te", te))}
    log["losses_at_236"]["worst_three_readings_pct"] = round(float(worst3(236)) * 100, 2)
    log["losses_at_165"] = {k: round(float(1 - revenue(v, 165) / Rs[k]) * 100, 2)
                            for k, v in (("ch", ch), ("ex", ex), ("te", te))}
    # left-limit worst cases are suprema: the adversary with q = p - 1/1000
    # comes within 0.001 of rho and never exceeds it
    for lab, lo_, hi_ in (("survey_COH_tc_te", tc, te), ("survey_inner_ch_ex", ch, ex)):
        n_ll = 0
        for p in sorted(set(lo_) | set(hi_)):
            sh, q, kd = guaranteed_share(lo_, hi_, p)
            if kd == "left-limit":
                reg = adversary_regret(lo_, hi_, p, Fr(p) - Fr(1, 1000), "below")
                assert 0 <= (1 - sh) - reg < Fr(1, 1000), (lab, p)
                n_ll += 1
        mm[lab]["left_limit_candidates_checked"] = n_ll
    print("losses at 236:", log["losses_at_236"], "at 165:", log["losses_at_165"])
    print("alt-demand reading below OPP:", alt["witness"]["reading_with_optimum_below_OPP"])

    out = os.path.join(RESULTS, "06_coldfix_additions.json")
    with open(out, "w") as f:
        json.dump(log, f, indent=2)
    for k, v in mm.items():
        print(k, "minimax price", v["minimax_price"], "worst-case forgone", v["worst_case_share_forgone_pct"], "%",
              "vs q", v["worst_case_comparison_price"], v["worst_case_kind"], "max g", v["max_g"], "max h", v["max_h"])
    for k, v in mm.items():
        print(k, "text rows:", v["text_rows"])
    print("crossings:", json.dumps(cr))
    print("overlap:", log["overlap"])
    print("ties:", block)
    print("alt demand:", alt)
    print("section 7:", log["section7_population_values"])
    print("prop 3 family:", fam)
    print("three-reading:", log["three_reading_minimax"])
    print("wrote results/06_coldfix_additions.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
