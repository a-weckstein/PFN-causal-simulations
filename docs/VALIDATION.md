# VALIDATION.md — reproduction of archived per-sim results (2026-07-04)

> **Addendum 2026-07-26 — what changed in the comparison targets since this
> validation ran.** The results below are a faithful record of the 2026-07-04
> runs and are not rewritten; three production changes since then affect how
> to read them:
>
> 1. **n=1200 backfill relabeling (2026-07-05).** §2's "EXACT after shifting
>    sim_id by one" was measured against the pre-remediation archive.
>    Production has since relabeled the backfill CSVs and rebuilt
>    `unified_sim_results_n1200.csv`, so those cells now match the current
>    archive directly — no shift needed (DEVIATIONS D1).
> 2. **TabPFN v3 cloud drift (measured 2026-07-24).** §3's offline agreement
>    (saved fits → this code's estimators, ipw/gcomp/aipw ≤ 2.7e-13) stands.
>    But it validates the *estimator path*, not fresh-API reproducibility:
>    the v3 endpoint drifts server-side across weeks, so a fresh API run
>    reproduces neither the archives nor this validation (DEVIATIONS D5
>    correction; `tmleR_extension_n1200_n5000_PLAN.md` REVISION v2).
> 3. **New production learners (2026-07-23→26).** `tabpfn_v3_api_tmleR`
>    (n=200, then n=1200) and `tabpfn_api_v3_regen` (n=1200, n=5000 in
>    flight) postdate this file; nothing here compares against them. This
>    code's `tabpfn_v3` TMLE rows correspond to `tabpfn_v3_api_tmleR`
>    per DEVIATIONS D5.

This codebase was validated against the production studies by re-running
simulations and comparing per-sim rows (estimate, SE, CI bounds, coverage)
against the archived canonical per-sim files:

- n=200:  `Naimi_v4.n200/results/combined_all_tracks_n200.csv`
- n=1200: `Naimi_streamlined.v4/results/unified_sim_results_n1200.csv`
- n=5000: `Naimi_streamlined.v4_n5000/results/all_results_combined_n5000.csv`

"EXACT" below means max |difference| ≤ ~5e-14 across all compared rows —
i.e. bit-level agreement up to CSV round-trip precision — with 100%
agreement on the `covered` indicator. All comparisons are at the default
truncation arm and include all four estimators (gcomp compared on estimate
only; it has no SE by design).

## 1. Datasets and folds

Stage-1 datasets were compared field-by-field (C, Z, A, Y, pi_true,
mu0_true, mu1_true, Y0, Y1, seed) against the production `_data_inputs/`
caches: **max |diff| = 0 at all three sample sizes** (n=200 sims 1–3,
n=1200 sims 1–3, n=5000 sims 1–2). The exported `fold_id` column matches
production's `generate_folds(n, K, seed = sim_seed*1000+7)` exactly.

## 2. R-track estimator rows

### n=200 (sims 1–3; 24 rows per cell = 3 sims × 2 scenarios × 4 estimators)

| Learners | NoCF | CF |
|---|---|---|
| parametric, gam, lasso, knn, mlp | EXACT | EXACT |
| ranger, xgboost, xgboost_naimi, bart | EXACT | EXACT |
| sl_balzer, sl_naimi_v1 | EXACT | EXACT |
| sl_default, sl_naimi_v2 (sims 1–2) | EXACT | EXACT |
| hal_discrete (sims 1–2) | EXACT | differs — archived rows are irreproducible draws (mclapply time+PID worker seeding; DEVIATIONS D8, ISSUES §9). Verified: production-style re-runs differ from themselves. This code now uses deterministic per-fold streams. |
| oracle, oracle_notrunc | EXACT | EXACT |

(sl_default validated on both scenarios; sl_naimi_v2 and hal_discrete on the
simple scenario plus sl_naimi_v2 complex NoCF — the scenario changes only
the covariate matrix, not the code path.)

### n=1200 (sims 1–3)

| Learners | NoCF | CF |
|---|---|---|
| parametric | EXACT (preliminary source) | differs — archived rows are `imported_from_v3_cf` (deleted v3 pipeline, its own fold partitions; not reproducible by any current code) |
| gam, knn, mlp, bart | EXACT | EXACT |
| lasso | differs — archived NoCF was unseeded (DEVIATIONS D1); CF EXACT | EXACT |
| xgboost, xgboost_naimi | EXACT (xgboost; preliminary source) | **EXACT after shifting sim_id by one** (both learners, tested independently) — archived rows come from `run_cf_backfill.R`, which mislabels sims (ISSUES §6). Unshifted max \|Δ estimate\| ≈ 2–2.7. |

The xgboost result generalizes: the backfill-sourced CF cells (xgboost,
xgboost_naimi, sl_default, sl_naimi_v1) reproduce exactly under the
corrected sim labeling, which is both the validation of this code and the
demonstration of the production off-by-one. The fifth backfilled learner,
hal_discrete, is additionally subject to the CF irreproducibility above, so
its rows cannot be matched under any labeling.

### n=5000 (sims 1–3)

| Learners | NoCF | CF |
|---|---|---|
| parametric | EXACT | EXACT |

(The n=5000 archived "HAL" is the simplified `hal` variant, deliberately not
implemented here — DEVIATIONS D6. Other n=5000 learners share the exact code
paths validated at n=200/n=1200 plus the same folds_by_n=5 setting validated
via parametric.)

## 3. TabPFN v3 track (offline, no API calls)

Production saved raw v3 nuisance fits for n=200 sims 1–50
(`Naimi_v4.n200/results/adaptive_smoke_n200/v3_raw_fits_n200.csv`, verified
in production to reproduce its own 200-sim run to ~1e-13). Feeding those
saved nuisances through THIS codebase's estimator path (stage 2b: truncate
to [0.025, 0.975] → the four R estimators) and comparing to archived v3
rows (sims 1–5, NoCF + CF):

| Estimator | max \|Δ estimate\| | Interpretation |
|---|---|---|
| ipw | 2.7e-13 | exact — R estimator ≡ production Python estimator |
| gcomp | 1.8e-15 | exact |
| aipw | 3.6e-15 | exact |
| tmle | 1.88 (NoCF), 0.13 (CF) | expected — DEVIATIONS D5: this code uses `tmle::tmle()` (logistic fluctuation) for every learner, production Python used a linear fluctuation. The largest gaps occur in NoCF sims with extreme in-sample propensities. |

Archived TabPFN TMLE rows are therefore NOT reproduced by design; all other
TabPFN estimator rows are.

The same check was run end-to-end through the actual `--stage 2b` script
(saved fits converted to the `tabpfn_v3_nuisance/` file format, sims 1–3,
48 rows): identical outcome — ipw/gcomp/aipw ≤ 2.5e-13, tmle max |Δ| 0.41
(NoCF) / 0.13 (CF).

## 4. Environment

R 4.5.2 (2025-10-31) on macOS (Darwin 24.5.0). Package versions during
validation: tmle 2.1.1, SuperLearner 2.0.40, sandwich 3.1.1, ranger 0.18.0,
xgboost 3.1.3.1, glmnet 4.1.10, gam 1.22.7, earth 5.3.5, nnet 7.3.20,
caret 7.0.1, dbarts 0.9.32, hal9001 0.4.6, ggplot2 4.0.1.

Exact reproduction of RNG-dependent learners assumes the same package
versions (a package changing its internal RNG consumption would break
bit-agreement without invalidating the statistics). Note xgboost proved
insensitive to R's RNG state in these tests (its CV folding does not draw
from R's stream at this version), while cv.glmnet, ranger, SuperLearner and
nnet/dbarts (via content seeds) do consume R RNG as expected.

## 5. How to re-run this validation

```bash
Rscript run.R --n 200 --stage 1 --sims 1:3
Rscript run.R --n 200 --stage 2 --sims 1:3          # all learners + oracles
# then compare results/n200/per_config/ against the archived per-sim file
# on (sim_id, scenario, learner, cross_fit, estimator).
```

## 6. Addendum 2026-08-15 — three-script packaging validation

The consolidated scripts (`01/02/03_*.R`, assembled by verbatim concatenation
of the module files this validation covered) were re-validated against the
modular pipeline at n=200, sims 1–2: stage-1 RDS/CSV byte-identical;
per-config rows for parametric/gam/lasso/knn/ranger/xgboost_naimi/oracle and
the TabPFN estimator track EXACT (max |Δ| = 0 on estimate/SE/CI/covered);
stage-3 summary metrics EXACT across all 128 cells. New `hal_n5000` learner
(D11) smoke-tested and verified deterministic across fork counts.
