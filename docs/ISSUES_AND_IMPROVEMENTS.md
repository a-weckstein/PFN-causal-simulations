# ISSUES_AND_IMPROVEMENTS.md — findings from the consolidation pass (2026-07-04)

Objective-2 deliverable: issues, inconsistencies, and improvement
opportunities in the production implementations
(`Naimi_streamlined.v4`, `Naimi_streamlined.v4_n5000`, `Naimi_v4.n200`)
observed while building and validating this clean codebase. Items already
tracked in `Fable_review_2026-07-02/` are referenced rather than re-litigated;
new observations are marked **[new]**.

## A. Issues that affect interpretation of archived numbers

1. **Unseeded historical NoCF runs at n=1200** (known). NoCF rows for
   stochastic learners at n=1200 cannot be exactly reproduced by any code.
   Impact: contrasts against those rows carry an extra (small) run-to-run
   randomness component. CF rows are unaffected (fold-seed pinned).
   Manuscript handling: nothing needed beyond the standard MC-error framing;
   exact-reproducibility claims should be limited to n=200/n=5000 and all CF
   cells.

2. **n=1200 truncation-sensitivity TMLE rows are mislabeled in effect**
   (known; family overview §gbound). Because the parents never passed
   `gbound`, the `less_aggressive` [0.005, 0.995] and `none` [0, 1] TMLE
   sensitivity rows at n=1200 were both effectively run at the package's
   internal floor [0.0204, 0.9796] and are numerically identical to each
   other. If any of those rows appear in the manuscript, relabel as
   "tmle-package adaptive truncation" or drop. Default-truncation rows are
   unaffected everywhere.

3. **Cross-track TabPFN comparability at n=1200/n=5000** (known). The cloud
   track ran on its own numpy-generated datasets: same DGP, different
   realizations, plus a different TMLE targeting algorithm (linear vs
   logistic fluctuation). Archived cross-track comparisons at those n are
   aggregate-only and TMLE comparisons are approximate. (This codebase
   removes both inconsistencies going forward — see DEVIATIONS D3–D5.)

4. **"HAL" at n=5000 is a different algorithm** (known). The n=5000 primary
   `hal` is a simplified single-fit variant standing in for `hal_discrete`.
   Cross-n "HAL" comparisons compare different estimators of the nuisance;
   state in methods (registry §3.2 wording is good).

5. **Tuning/CF folds 10 vs 5 across sample sizes** (known, accepted).
   Config-driven, deliberate; state once in methods.

## B. New findings from this pass **[new]**

6. **Off-by-one sim labeling in the n=1200 CF backfill (data-integrity issue).
   REMEDIATED 2026-07-05** — AW chose option (a)+: backfill CSVs relabeled
   (sim_id −1, orphan dropped, originals archived), the runner's seed rule
   fixed, canonical sim 200 backfilled through the fixed runner (verified vs
   this codebase to 4e-15 for xgboost/xgboost_naimi), unified file +
   dashboard + n=1200 tables rebuilt; aggregate shifts confined to the 40
   backfilled CF cells (RMSE ≤ 0.011). Original finding below for the record.
   `Naimi_streamlined.v4/analysis/run_cf_backfill.R` derives
   `seed <- STARTING_SEED + sim_id - 1` and regenerates data from that seed,
   whereas the canonical pipeline uses `seed = starting_seed + sim_id`
   (sims 1..200 → seeds 2..201, cached in `_data_inputs/`). The backfill
   therefore computed each row labeled `sim_id = k` on the dataset the rest
   of the study calls **sim k−1** (and its `sim_id = 1` uses seed 1, a
   dataset that exists nowhere else in the study).
   - **Evidence:** this codebase reproduces the backfill's xgboost CF rows to
     5e-15 after shifting sim_id by one, and does NOT reproduce them
     unshifted (max |Δ estimate| ≈ 2.7 across sims 1–3); with the correct
     convention it reproduces every 2026-06-22/23-campaign cell exactly.
   - **Blast radius:** in the canonical per-sim file
     `results/unified_sim_results_n1200.csv`, ALL CF rows for **xgboost,
     xgboost_naimi, sl_default, sl_naimi_v1, hal_discrete** (5 learners × 2
     scenarios × 200 sims × 4 estimators = 8,000 rows) are sourced from
     `cf_backfill_n1200`.
   - **Impact:** aggregate metrics (bias, RMSE, coverage over 200 sims) are
     statistically unaffected — the rows are still 200 valid draws from the
     DGP (199 of the 200 datasets are shared with the canonical set, one is
     swapped). But **within-sim paired contrasts are misaligned** for those
     cells: CF-vs-NoCF deltas within those learners, learner-vs-learner
     per-sim comparisons involving them, and joins to the oracle rows all
     compare estimates computed on different datasets under the same sim_id.
   - **Known affected consumer:** `AW_Result_formatting/helpers/
     naimi_figures_helpers.R` `tie_mode="paired"` (added 2026-07-03) computes
     learner-vs-learner Monte-Carlo error as sd(per-sim difference)/sqrt(n),
     keyed by sim_id and premised on "learners share the same sim draws".
     For n=1200 CF panels involving the five backfilled learners, the pairs
     are misaligned: the paired MCSE loses the shared-draw correlation
     (windows too wide or simply wrong), so paired tie/tier assignments in
     those ranking panels are unreliable. Marginal (unpaired) metrics and
     whiskers are fine.
   - **Remediation options:** (a) relabel — subtract 1 from sim_id in
     cf_backfill-sourced rows and drop backfill sim 1 (restores pairing for
     199 sims; cheapest, exact; also valid for hal_discrete, whose rows are
     additionally irreproducible per issue 9 but were still computed on the
     seed-k datasets); (b) re-run those five CF cells through the
     standard stage-2 path (xgboost/xgboost_naimi cheap;
     sl_default/sl_naimi_v1/hal_discrete expensive); or (c) treat those
     cells as an unpaired alignment group in the paired-ties machinery
     (it already supports cross-group fallback to independent MCSE) and
     restrict them to aggregate-only claims.

7. **Python-track `seed` column is off by one relative to the R convention.**
   The TabPFN notebooks/headless runners label `seed = starting_seed +
   sim_id − 1` (= sim_id), while R-track rows carry `seed = sim_id + 1`.
   At n=200 the v3 track reads the R-exported datasets, so the DATA are
   correct — only the recorded seed label differs. Any cross-track join must
   key on `sim_id`, never on `seed`. (The same convention inside
   `run_cf_backfill.R` is what caused issue 6, where data were regenerated
   rather than read.)

8. **n=1200 unified CF rows are three different vintages.** Canonical CF
   rows at n=1200 come from: the 2026-06-22/23 standalone campaign
   (gam, knn, lasso, mlp, bart — bit-reproducible, verified), the CF
   backfill (issue 6), and `imported_from_v3_cf` (parametric, ranger,
   sl_balzer, sl_naimi_v2, tabpfn — same datasets per sim, verified, but
   fold partitions from the deleted v3 pipeline, so not reproducible and
   their CF noise differs from the v4 convention). Cross-learner per-sim CF
   comparisons at n=1200 therefore mix fold conventions; within-directory
   pairing is fully clean only at n=200 and n=5000.

   **Re-verified and extended 2026-07-28.** The "same datasets per sim"
   clause above was the only correct statement of this fact anywhere in the
   project — the production docs (`Naimi_streamlined.v4/_reference/handoff.md`
   seed-alignment table, `results/imported_from_v3_nocf/README.md`, that
   directory's `CLAUDE.md`, `_reference/handoff_oracle.md`) all asserted the
   opposite, tracing to the 2026-06-10 review §1.3, which inferred a
   different v3 seed scheme from the v3 codebase being unavailable rather
   than from the archived numbers. All four have now been corrected. The
   evidence, which also **extends the finding to the NoCF imports** (issue
   1's cohort — previously untested):
   - `imported_from_v3_nocf`: `parametric`/simple is deterministic and
     recomputes from the current `_data_inputs/*.rds` cache over **all 200
     sims × 3 estimators to max abs(Δ) = 4.9e-13**.
   - `imported_from_v3_cf`: paired cor(v3-CF, v4-NoCF) over 200 sims =
     0.79–0.998 (parametric, ranger, sl_balzer, sl_naimi_v2, tabpfn) vs
     −0.006…+0.008 under shuffled `sim_id`; and `parametric` CF recomputed
     under the **v4** fold rule differs from the archive by only 0.005–0.13
     — fold noise against a between-sim SD of 0.63, not a different draw.
   - Issue 6's remediation independently confirmed: xgboost CF sims 1–2
     reproduce the relabeled backfill at the same `sim_id` to
     max abs(Δ) = 0.0e+00.

   Re-runnable: `Naimi_streamlined.v4/analysis/verify_v3_import_alignment.R`.
   **Net effect on this issue:** the *data* axis at n=1200 is clean for every
   R-track learner (so per-sim pairing among them is valid, including against
   the R oracles); what genuinely remains is the fold-convention mixing
   described above plus issue 1's unseeded-NoCF irreproducibility. The
   Python TabPFN tracks remain on different realizations entirely
   (DEVIATIONS D3) — unchanged by any of this.

9. **hal_discrete CF rows are not reproducible (any of them, by anyone).**
   The CF fold loop runs under `mclapply` with the default
   `mc.set.seed = TRUE` under Mersenne-Twister, which re-seeds each forked
   fold worker from time + process ID. Verified empirically: two runs of the
   same (sim, config) with identical upstream seeding give different
   hal_discrete CF nuisances (max |Δ pihat| ≈ 0.17, max |Δ mu0hat| ≈ 15 on
   n=200 sim 1) — production's own archived hal_discrete CF rows are
   run-specific draws. Affected: hal_discrete CF cells at n=200 and n=1200
   (including the backfill). The fits are statistically valid; only
   bit-reproducibility is lost, so aggregate conclusions stand, but exact
   re-runs will not match and per-sim rows cannot be regenerated for audit.
   This codebase fixes it with deterministic L'Ecuyer-CMRG per-fold streams
   (DEVIATIONS D8) — verified identical across runs and core counts.

10. **Learner-internal CV seeds are shared with the fit stream.** For
   RNG-using learners, the internal CV fold draws and the model fits consume
   one stream seeded per (sim, path). Consequence: adding/removing an RNG
   call inside any learner silently changes that learner's downstream draws
   (but no other learner's, since each (learner, sim, path) re-seeds).
   This is why validation here ported learner internals verbatim. Any future
   edit inside a learner invalidates bit-reproducibility for that learner
   only — re-validate against archived rows after touching one.

11. **`covered` semantics for gcomp.** G-comp has no SE by design, so its
   coverage is NA. Production downstream guards handle this, but any ad-hoc
   analysis that computes `mean(covered)` with `na.rm=TRUE` over estimator
   subsets including gcomp silently averages over the other estimators only.
   The clean summary reports `coverage = NA` with an explicit `mcse_coverage
   = NA` for such cells.

12. **IPW SE convention.** The HC sandwich SE from the weighted `lm` treats
   the estimated weights as fixed (no correction for propensity estimation).
   This is the common convention (and matches Naimi et al.), generally
   conservative for the ATE with estimated weights; fine, but worth one
   methods sentence.

13. **Timing columns are not comparable across learners.** `nuisance_time`
    is CPU (parent + children) and thus counts parallel fold work fully
    (hal_discrete CF), while single-threaded learners report wall ≈ CPU.
    Production added `nuisance_time_wall_serial_eq` for this; if runtime
    comparisons appear in the manuscript, use per-learner like-for-like
    columns (or re-run the dedicated timing harness), not raw
    `nuisance_time`.

## C. Improvement opportunities (not implemented as fixes to production)

14. **G-comp bootstrap SE.** The gcomp arm could gain a bootstrap CI cheaply
    for the fast learners; it was consciously skipped in production. Would
    fill the one NA column of the results grid.

15. **Parallelize over sims, not just HAL folds.** Stage 2 is serial in sims
    within a config. The per-sim unit is embarrassingly parallel and the
    seeding is already per-sim, so `mclapply` over sims (with per-worker
    package loads) would cut wall time roughly by the worker count for the
    expensive learners. Kept serial here for simplicity and identical RNG
    behavior.

16. **Sandwich-free MC-SE reporting.** The clean stage 3 adds Monte-Carlo
    SEs for bias and coverage (production Naimi dashboards did not report
    them; the ACIC side did). Recommend quoting them in the manuscript
    tables (200 sims → coverage MC-SE ≈ 1.5 pp at 95%).

17. **Environment pinning.** No production directory pins R package
    versions; `tmle` 2.x behavior (gbound semantics) and `hal9001` internals
    matter. `renv::snapshot()` (or at minimum `sessionInfo()` capture into
    the results directory) is recommended when freezing manuscript numbers.
    This codebase prints package versions in `docs/VALIDATION.md` for the
    validation runs.

## D. Validation-run findings

See `docs/VALIDATION.md`. Summary: datasets and folds reproduce production
byte-identically at all three n; per-sim estimator rows reproduce production
exactly (max |diff| ~1e-14, CSV round-trip precision) for every learner
validated at n=200 (seeded throughout) and for CF cells at n=1200; n=1200
NoCF rows of stochastic learners differ as expected (issue A1).
