# PSM non-identification

Code, data transcription and results for a paper on what the Van Westendorp
Price Sensitivity Meter can and cannot identify (paper submitted to a journal
and under review; the reference will be added when it is published).

The paper constructs an explicit witness: two readings of one respondent
population that produce identical Van Westendorp PSM curves (hence the same
Optimal Price Point) while their revenue-optimal prices differ, so no
functional of the PSM output can recover the optimal price. The witness is
certified by exact-rational verification, extended to continuous populations
by Monte Carlo, and complemented by a characterization of the set of prices
that some coherent reading makes optimal, a positive result on when a
declared measurement equation restores identification, and a reanalysis of
a published 52-respondent survey.

## Repository layout

```
scripts/   Verification pipeline (Python 3): 00 to 06 plus the shared psm_core module
data/      The transcription of the published survey table used in the paper
results/   Every result file and figure cited in the paper
```

## Data sources

One external dataset: `data/dmitriev2019_psm_n52.csv` is a transcription of
Table 1 of Yu.G. Dmitriev, Zh.N. Zenkova and A.G. Zenkov (2019), "Statistical
properties of price sensitivity meter results", Tomsk State University
Journal of Control and Computer Science 47: 30-40,
https://doi.org/10.17223/19988605/47/4 (the complete respondent-level answers
of a real PSM survey of 52 organizations, published by the authors in
rescaled conventional units). The file header records the provenance; script
04 certifies the transcription by reproducing the article's four published
price points with the authors' own estimator. Everything else is generated
by the scripts from explicit distributions with fixed seeds. No APIs or keys
are involved.

`data/psm_package_vignette_sim_n250.csv` is SIMULATED data (the R package
pricesensitivitymeter's vignette example, reproduced bit-exactly by script
04); it is regenerated on every run.

## Reproduction

Requirements: Python 3 with `numpy` (and `matplotlib`, with Pillow for the
TIFF copies, for the figures in scripts 02 and 04); script 05 is pure
Python. All scripts are deterministic given their fixed seeds and regenerate
the results byte-identically. Run from `scripts/`:

```
python 00_anchor.py               # correctness anchor: recovers the hand-solved
                                  # uniform example (OPP 8, IPP 8, PMC 7, PME 9)
python 01_nonidentification.py    # builds the discrete witness and certifies the
                                  # theorem (01_counterexample.json, 01_curves.csv)
python 02_monte_carlo.py          # both continuous experiments (02_monte_carlo.json,
                                  # 02_figure.pdf/.eps/.png/.tif)
python 03_witness_certificate.py  # re-certifies the witness on a full-support grid
                                  # and by exact atom enumeration; computes the
                                  # positive-result quantities
                                  # (03_witness_certificate.json, 03_curves_full.csv)
python 04_empirical_range.py      # the published survey: transcription certificate,
                                  # curves, price points, per-reading revenue optima,
                                  # losses at the published OPP, interval-attainment
                                  # certificate (04_empirical_range.json and figure)
python 05_jrpm_additions.py       # exact rational arithmetic: the identified set of
                                  # coherent optima (both datasets, three bands), the
                                  # cost table and minimax relative-regret price,
                                  # the pooled median check (05_jrpm_additions.json)
python 06_coldfix_additions.py    # exact rational arithmetic: the worst-case share of
                                  # revenue forgone over a whole band and the band-wide
                                  # minimax relative-regret price (each value certified
                                  # by an explicit band member and by random readings),
                                  # the crossing convention and height, the alternative
                                  # demand check, the population values of the
                                  # continuous example (06_coldfix_additions.json)
```

The verification logic lives in `psm_core.py`; script 00 validates it against
an analytically solved example before anything else depends on it. The witness
generator in script 01 is parameterized so a skeptical reader can construct
further counterexamples. Figures are black and white (vector PDF and EPS, plus
600 dpi PNG and TIFF) since 2026-11-07; the numerical outputs did not change.

Script 04 keeps its July names: its `identified_range` is the paper's focal
range (the optima under the bargain and walk-away readings), not the
identified set, which script 05 computes (`identified_sets_published_survey`).
Its console headline prices at the recomputed crossing 288; the paper's
losses at the published 290 (37.8 and 1.7 per cent) are under
`published_OPP_290` in `04_empirical_range.json`. Script 05's
`minimax_regret_price_focal_readings` is the paper's minimax relative-regret
price: the price with the smallest worst-case share of attainable revenue
forgone across the declared readings.

## License

MIT, see `LICENSE`.
