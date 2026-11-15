"""
05_jrpm_additions.py - exact-arithmetic results added for the JRPM version of
the paper (2026-11-07). Pure Python (fractions, csv, json); no numpy.

What it computes, all in exact rational arithmetic:

  1. The identified set of coherent revenue optima (Proposition 1 of the JRPM
     version). For a band lo <= W <= hi (lo, hi two answer columns) write
         g(p) = p * Pr(lo >= p)      and      h(q) = q * Pr(hi >= q).
     A price q is the revenue optimum (smallest maximizer) of some coherent
     environment with the given marginals if and only if
         h(q) > g(p) for every p < q   and   h(q) >= g(p) for every p > q.
     Since g is increasing between the atoms of lo and jumps down just after
     each atom, and q*Pr(lo >= q) <= h(q) by cone-compatibility, the two
     conditions reduce to tests against the atoms of lo. The set is a finite
     union of intervals with rational endpoints; this script computes it,
     then re-checks it by brute force at thousands of rational points, and
     certifies membership constructively (the reading W_i = q when
     lo_i <= q <= hi_i, else W_i = lo_i, has exact optimum q).
     Computed for: the published survey of Dmitriev, Zenkova and Zenkov
     (2019) under the bands [tc, te], [ch, ex] and [ch, te]; the witness
     population of Table 1 under [tc, te] and [ch, ex]; the two-point
     'special marginals' example of Section 6.

  2. The cost table of Section 9.3: share of attainable revenue forgone at
     251, 253, 257, 290 and 301 under the bargain (W = ch), expensive
     (W = ex) and walk-away (W = te) readings, and the minimax-regret price
     (smallest worst-case loss across the declared readings).

  3. The W = tc reading's optimum (footnote of Section 8).

  4. The OPP as a pooled median (Section 6, after Example 1): the 52nd and
     53rd of the 104 pooled too-cheap and too-expensive answers, and the
     interval on which the crossing equation Pr(tc < p) + Pr(te <= p) = 1
     holds.

  5. Small checks used in the text: the height of the OPP crossing (2/52,
     Remark 2), the number of walk-away answers in the far tail of the
     identified set, Remark 1's epsilon margins, and the W = tc = te uniform
     example in which the OPP equals the optimum (used to word the claim
     'need not be the optimum').

Input:  data/dmitriev2019_psm_n52.csv (sha256 checked; the transcription
        certified by script 04).
Output: results/05_jrpm_additions.json
Run:    python3 scripts/05_jrpm_additions.py
"""
import csv
import hashlib
import json
import os
import sys
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)

CSV = os.path.join(DATA, "dmitriev2019_psm_n52.csv")
CSV_SHA256 = "45c4edca85aecd669d816bc3e81c0e5f91edf4a89cf9564316d8c13fddd96337"


# ---------------------------------------------------------------------------
# exact helpers on a finite population given as a list of values (equal
# weights) -- weighted populations are expanded to integer multiplicities
# ---------------------------------------------------------------------------
def surv(xs, p):
    """Pr(X >= p), exact."""
    return Fr(sum(1 for x in xs if x >= p), len(xs))


def revenue(xs, p):
    """R(p) = p * Pr(W >= p), exact."""
    return Fr(p) * surv(xs, p)


def optimum(xs):
    """Smallest maximizer of p * Pr(W >= p) by exact atom enumeration
    (the maximum sits at an atom: proof of Theorem 1)."""
    best = None
    for a in sorted(set(xs)):
        r = revenue(xs, a)
        if best is None or r > best[1]:
            best = (Fr(a), r)
    return best


def identified_set(lo, hi):
    """Union of intervals of prices q with
         q*Pr(hi>=q) > a*Pr(lo>=a) for all atoms a < q of lo, and
         q*Pr(hi>=q) >= a*Pr(lo>=a) for all atoms a > q of lo.
    Returns a list of (left, left_closed, right, right_closed) with Fraction
    endpoints, merged, plus max_p g(p)."""
    lo_atoms = sorted(set(lo))
    hi_atoms = sorted(set(hi))
    g = {a: revenue(lo, a) for a in lo_atoms}
    gmax = max(g.values())

    def g_below(q):      # max of g over atoms strictly below q (0 if none)
        vals = [g[a] for a in lo_atoms if a < q]
        return max(vals) if vals else Fr(0)

    def g_above(q):      # max of g over atoms strictly above q (0 if none)
        vals = [g[a] for a in lo_atoms if a > q]
        return max(vals) if vals else Fr(0)

    def member(q):
        hq = revenue(hi, q)
        return hq > g_below(q) and hq >= g_above(q)

    # breakpoints: every atom of either column; on the open interval between
    # consecutive breakpoints, Pr(hi >= q), g_below and g_above are constant.
    bps = sorted(set(Fr(x) for x in lo_atoms + hi_atoms))
    pieces = []
    left = Fr(0)
    for b in bps:
        pieces.append((left, b))
        left = b
    segs = []
    for (x0, x1) in pieces:
        mid = (x0 + x1) / 2
        c = surv(hi, mid)                      # Pr(hi >= q) on (x0, x1]
        if c == 0:
            # h = 0 on this piece; q can be a member only if g_below(q) < 0,
            # impossible: skip (the endpoint x1 is tested separately below)
            pass
        else:
            gb, ga = g_below(mid), g_above(mid)
            # conditions: c*q > gb  and  c*q >= ga  on the open interval
            lo_strict = gb / c          # q > lo_strict
            lo_weak = ga / c            # q >= lo_weak
            a = max(x0, lo_strict, lo_weak)
            # closedness at a: if a == x0 the interval is open at x0 (x0 is
            # the previous piece's right end, tested on its own); if a came
            # from lo_weak only (and a > lo_strict) it is closed
            if a < x1:
                left_closed = (a > x0) and (a == lo_weak) and (a > lo_strict)
                segs.append([a, left_closed, x1, member(x1)])
        # right endpoint tested exactly (it may hold even if the interior does not)
        if member(x1):
            if not (segs and segs[-1][2] == x1 and segs[-1][3]):
                segs.append([x1, True, x1, True])
    # beyond the last atom of hi, Pr(hi >= q) = 0: nothing
    # merge touching segments
    merged = []
    for s in segs:
        if merged:
            p = merged[-1]
            if p[2] == s[0] and (p[3] or s[1]):
                p[2], p[3] = s[2], s[3]
                continue
            if p[2] == s[2] and p[0] <= s[0]:
                continue
        merged.append(list(s))
    # brute-force re-check at many rational points
    test_points = set()
    for s in merged:
        test_points.update([s[0], s[2], (s[0] + s[2]) / 2])
    for i in range(len(bps) - 1):
        test_points.update([bps[i], (bps[i] + bps[i + 1]) / 2,
                            bps[i] + (bps[i + 1] - bps[i]) / 7])
    top = bps[-1]
    for k in range(1, 2001):
        test_points.add(top * Fr(k, 2000))
    for q in test_points:
        if q <= 0:
            continue
        inside = any((s[0] < q < s[2]) or (q == s[0] and s[1]) or (q == s[2] and s[3])
                     for s in merged)
        assert inside == member(q), f"identified-set mismatch at q={q}"
    return merged, gmax


def seg_json(segs):
    return [{"left": str(a), "left_float": float(a), "left_closed": lc,
             "right": str(b), "right_float": float(b), "right_closed": rc}
            for a, lc, b, rc in segs]


def seg_text(segs):
    return " U ".join(f"{'[' if lc else '('}{float(a):.4g}, {float(b):.4g}{']' if rc else ')'}"
                      for a, lc, b, rc in segs)


def certify_members(lo, hi, segs, qs):
    """For each q, the reading W_i = q when lo_i <= q <= hi_i, else W_i = lo_i
    (rows paired as given) must have exact optimum q; for q outside the set the
    bound inequality must fail."""
    out = []
    for q in qs:
        q = Fr(q)
        inside = any((s[0] < q < s[2]) or (q == s[0] and s[1]) or (q == s[2] and s[3])
                     for s in segs)
        W = [q if (l <= q <= h) else Fr(l) for l, h in zip(lo, hi)]
        assert all(l <= w <= h for l, w, h in zip(lo, W, hi))
        p_star, r_star = optimum(W)
        if inside:
            assert p_star == q, (q, p_star)
        else:
            assert p_star != q, (q, p_star)
        out.append({"q": str(q), "in_set": inside, "construction_optimum": str(p_star),
                    "construction_revenue": str(r_star)})
    return out


def main():
    raw = open(CSV, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == CSV_SHA256, f"data/dmitriev2019_psm_n52.csv modified (sha256 {sha})"
    rows = [r for r in csv.reader(raw.decode().splitlines())
            if r and not r[0].startswith("#") and r[0] != "respondent"]
    assert len(rows) == 52
    tc = [int(r[1]) for r in rows]
    ch = [int(r[2]) for r in rows]
    ex = [int(r[3]) for r in rows]
    te = [int(r[4]) for r in rows]
    assert all(a <= b <= c <= d for a, b, c, d in zip(tc, ch, ex, te))
    log = {"data_file": "data/dmitriev2019_psm_n52.csv", "sha256": sha, "n": 52}

    # ---- 3. readings and the W = tc optimum ------------------------------
    readings = {}
    for name, xs in (("tc", tc), ("ch", ch), ("ex", ex), ("te", te)):
        p, r = optimum(xs)
        readings[name] = {"p_star": str(p), "revenue_at_p_star": str(r),
                          "revenue_float": float(r)}
    assert readings["ch"]["p_star"] == "251" and readings["ex"]["p_star"] == "253"
    assert readings["te"]["p_star"] == "301" and readings["tc"]["p_star"] == "125"
    assert Fr(readings["tc"]["revenue_at_p_star"]) == Fr(4625, 52)
    log["readings_exact"] = readings

    # ---- 1. identified sets ----------------------------------------------
    sets = {}
    for label, (lo, hi) in (("COH_tc_te", (tc, te)), ("inner_ch_ex", (ch, ex)),
                            ("ch_te", (ch, te))):
        segs, gmax = identified_set(lo, hi)
        sets[label] = {"max_p_g": str(gmax), "max_p_g_float": float(gmax),
                       "set": seg_json(segs), "set_text": seg_text(segs)}
    # the published survey: expected values
    coh = sets["COH_tc_te"]["set"]
    assert Fr(coh[0]["left"]) == Fr(4625, 52) and Fr(coh[0]["right"]) == 690
    assert Fr(coh[1]["left"]) == Fr(4625, 6) and Fr(coh[1]["right"]) == 773
    assert coh[1]["left_closed"] is False and coh[1]["right_closed"] is True
    inner = sets["inner_ch_ex"]["set"]
    assert len(inner) == 1 and Fr(inner[0]["left"]) == Fr(753, 4) and Fr(inner[0]["right"]) == 354
    segsC, _ = identified_set(tc, te)
    sets["COH_tc_te"]["constructive_certificates"] = certify_members(
        tc, te, segsC, [Fr(4625, 52), 125, 251, 290, 301, 690, 700, 770, 771, 773, 800])
    segsI, _ = identified_set(ch, ex)
    sets["inner_ch_ex"]["constructive_certificates"] = certify_members(
        ch, ex, segsI, [Fr(753, 4), 251, 253, 290, 354, 355, 400])
    # the inner-band reading that makes the published 290 optimal
    W290 = [290 if (c <= 290 <= e) else c for c, e in zip(ch, ex)]
    n_window = sum(1 for c, e in zip(ch, ex) if c <= 290 <= e)
    p290, r290 = optimum(W290)
    assert p290 == 290 and n_window == 20 and r290 == Fr(5945, 26)
    sets["inner_ch_ex"]["reading_making_290_optimal"] = {
        "construction": "W_i = 290 for the respondents with ch_i <= 290 <= ex_i, W_i = ch_i otherwise",
        "n_respondents_in_window": n_window, "optimum": str(p290),
        "revenue_at_290": str(r290), "revenue_float": float(r290)}
    # far tail: walk-away answers in (690, 773]
    sets["COH_tc_te"]["walk_away_answers_at_or_above_773"] = sum(1 for x in te if x >= 773)
    sets["COH_tc_te"]["walk_away_answers_above_690"] = sum(1 for x in te if x > 690)
    sets["COH_tc_te"]["smallest_walk_away_answer"] = min(te)
    log["identified_sets_published_survey"] = sets

    # the witness population of Table 1 (weights .35/.25/.25/.15 = 7/5/5/3 of 20)
    wtc = [4] * 7 + [8] * 5 + [12] * 5 + [16] * 3
    wch = [7] * 7 + [12] * 5 + [18] * 5 + [24] * 3
    wex = [10] * 7 + [16] * 5 + [24] * 5 + [32] * 3
    wte = [14] * 7 + [22] * 5 + [32] * 5 + [44] * 3
    wsets = {}
    for label, (lo, hi) in (("COH_tc_te", (wtc, wte)), ("inner_ch_ex", (wch, wex))):
        segs, gmax = identified_set(lo, hi)
        wsets[label] = {"max_p_g": str(gmax), "max_p_g_float": float(gmax),
                        "set": seg_json(segs), "set_text": seg_text(segs)}
    wc = wsets["COH_tc_te"]["set"]
    assert Fr(wc[0]["left"]) == Fr(26, 5) and Fr(wc[0]["right"]) == 32
    assert Fr(wc[1]["left"]) == Fr(104, 3) and Fr(wc[1]["right"]) == 44
    assert wc[1]["left_closed"] is False and wc[1]["right_closed"] is True
    W44 = [4] * 7 + [8] * 5 + [12] * 5 + [44] * 3
    p44, r44 = optimum(W44)
    assert p44 == 44 and r44 == Fr(33, 5)
    assert all(a <= w <= b for a, w, b in zip(wtc, W44, wte))
    wsets["COH_tc_te"]["reading_making_44_optimal"] = {
        "W_by_segment": [4, 8, 12, 44], "optimum": "44", "revenue": str(r44),
        "revenue_float": float(r44),
        "g_at_8": str(revenue(wtc, 8)),
        "h_on_32_44": "0.15 q, which exceeds 5.2 only for q > 104/3"}
    wsets["COH_tc_te"]["p_star_A_12_in_set"] = any(
        Fr(s["left"]) <= 12 <= Fr(s["right"]) for s in wc)
    wsets["COH_tc_te"]["p_star_B_22_in_set"] = any(
        Fr(s["left"]) <= 22 <= Fr(s["right"]) for s in wc)
    log["identified_sets_witness_table1"] = wsets

    # the two-point 'special marginals' example of Section 6
    segs, gmax = identified_set([1, 100], [2, 100])
    assert len(segs) == 1 and segs[0][0] == 100 and segs[0][2] == 100
    log["special_marginals_example"] = {
        "F_tc": "mass 1/2 at 1 and at 100", "F_te": "mass 1/2 at 2 and at 100",
        "set_text": seg_text(segs), "note": "every coherent reading prices at 100"}

    # ---- 2. cost table and minimax regret ---------------------------------
    R_star = {k: optimum(v)[1] for k, v in (("ch", ch), ("ex", ex), ("te", te))}
    cands = {251: "bargain optimum", 253: "expensive optimum",
             257: "minimax-regret price", 290: "published OPP", 301: "walk-away optimum"}
    table = {}
    for p, label in cands.items():
        losses = {k: 1 - revenue(v, p) / R_star[k] for k, v in (("ch", ch), ("ex", ex), ("te", te))}
        table[str(p)] = {"label": label,
                         **{f"loss_{k}": str(v) for k, v in losses.items()},
                         **{f"loss_{k}_pct": round(float(v) * 100, 2) for k, v in losses.items()},
                         "worst_case_focal_ch_te_pct": round(float(max(losses["ch"], losses["te"])) * 100, 2),
                         "worst_case_all_three_pct": round(float(max(losses.values())) * 100, 2)}
    # minimax over integer prices (focal readings and all three), plus a fine check
    def worst(p, keys):
        return max(1 - revenue(dict(ch=ch, ex=ex, te=te)[k], p) / R_star[k] for k in keys)
    top = max(te)
    mm_focal = min(range(1, top + 1), key=lambda p: (worst(p, ("ch", "te")), p))
    mm_all = min(range(1, top + 1), key=lambda p: (worst(p, ("ch", "ex", "te")), p))
    assert mm_focal == 257 and mm_all == 257
    # fine rational grid near the minimum (step 1/100 on [240, 310])
    fine = [Fr(240) + Fr(k, 100) for k in range(0, 7001)]
    mm_fine = min(fine, key=lambda p: (worst(p, ("ch", "te")), p))
    assert mm_fine == 257, mm_fine
    log["cost_table_published_survey"] = {
        "R_star": {k: str(v) for k, v in R_star.items()},
        "rows": table,
        "minimax_regret_price_focal_readings": mm_focal,
        "minimax_regret_worst_case_focal_pct": round(float(worst(257, ("ch", "te"))) * 100, 2),
        "minimax_regret_price_all_three_readings": mm_all,
        "fine_grid_check": "step 0.01 on [240, 310]: the minimax price is 257 (the walk-away loss 1 - p/R*_te falls to 257 and jumps at the first walk-away answer 257)",
        "ratio_worst_case_290_over_257": round(float(worst(290, ("ch", "te")) / worst(257, ("ch", "te"))), 2),
    }

    # ---- 4. the OPP as a pooled median --------------------------------------
    pooled = sorted(tc + te)
    cross = [p for p in range(min(tc), max(te) + 1)
             if Fr(sum(1 for x in tc if x < p), 52) + Fr(sum(1 for x in te if x <= p), 52) == 1]
    # equality holds on an interval of real p; integer witnesses and the open ends:
    log["pooled_median"] = {
        "pooled_52nd": pooled[51], "pooled_53rd": pooled[52],
        "dmitriev_estimator_is_53rd_of_104": pooled[52],
        "crossing_equation_Pr_tc_lt_p_plus_Pr_te_le_p_equals_1_at_integer_p": cross,
        "crossing_interval_open_ends": [288, 290],
        "note": "Pr(tc<p)+Pr(te<=p) = 1 on the open interval (288, 290): the left end 288 is the recomputed OPP, the right end 290 the study's estimator; both are sample medians of the 104 pooled answers (52nd value 288, 53rd value 290)",
    }
    assert pooled[51] == 288 and pooled[52] == 290 and cross == [289]
    assert Fr(sum(1 for x in tc if x < Fr(289)), 52) + Fr(sum(1 for x in te if x <= Fr(289)), 52) == 1
    assert Fr(sum(1 for x in tc if x < Fr(5779, 20)), 52) + Fr(sum(1 for x in te if x <= Fr(5779, 20)), 52) == 1  # 288.95

    # ---- 5. small checks -----------------------------------------------------
    # OPP height on (288, 290): Pr(tc >= p) = 2/52 (answers 290 and 301)
    log["opp_height_check"] = {"Pr_tc_ge_289": str(surv(tc, 289)),
                               "tc_answers_at_or_above_289": sorted(x for x in tc if x >= 289),
                               "steps_to_zero": 2}
    assert surv(tc, 289) == Fr(2, 52)
    # Remark 1 margins with W_B = te - eps
    def margins(eps):
        WB = [x - eps for x in wte]
        R22 = revenue(WB, 22 - eps); R14 = revenue(WB, 14 - eps); R32 = revenue(WB, 32 - eps)
        return R22 - R14, R22 - R32
    for eps in (Fr(1), Fr(3), Fr(59, 10)):
        m1, m2 = margins(eps)
        assert m1 == Fr(3, 10) + Fr(35, 100) * eps and m2 == Fr(15, 10) - Fr(25, 100) * eps
        assert optimum([x - eps for x in wte])[0] == 22 - eps
    log["remark1_epsilon_margins"] = {"margin_over_14_minus_eps": "0.3 + 0.35 eps",
                                      "margin_over_32_minus_eps": "1.5 - 0.25 eps",
                                      "checked_eps": ["1", "3", "5.9"], "optimum": "22 - eps"}
    # uniform example where OPP = p*: W = tc = te ~ U(0, 2); OPP = median = 1; R(p) = p(2-p)/2 max at 1
    log["opp_equals_optimum_example"] = {
        "law": "W = tc = te uniform on [0, 2]", "OPP_median": 1,
        "revenue": "p (2 - p) / 2, maximized at p = 1",
        "use": "shows the OPP can equal the optimum; the paper says it need not and cannot be shown to"}

    out = os.path.join(RESULTS, "05_jrpm_additions.json")
    with open(out, "w") as f:
        json.dump(log, f, indent=2)
    print("identified set, published survey, COH:", sets["COH_tc_te"]["set_text"])
    print("identified set, published survey, inner band:", sets["inner_ch_ex"]["set_text"])
    print("identified set, witness, COH:", wsets["COH_tc_te"]["set_text"])
    print("cost table:", {p: (r["loss_ch_pct"], r["loss_ex_pct"], r["loss_te_pct"]) for p, r in table.items()})
    print("minimax-regret price:", mm_focal, "worst case", log["cost_table_published_survey"]["minimax_regret_worst_case_focal_pct"], "%")
    print("pooled 52nd/53rd:", pooled[51], pooled[52])
    print("wrote results/05_jrpm_additions.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
