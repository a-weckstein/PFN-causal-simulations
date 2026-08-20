# DEVIATIONS.md — where this code differs from the production implementations

The archived study was produced by three separate directories
(`Naimi_streamlined.v4` at n=1200, `Naimi_streamlined.v4_n5000`,
`Naimi_v4.n200`) whose fix sets and conventions drifted apart over time.
This codebase implements the single most correct/consistent version of each
divergent choice. Each entry states what production did, what this code does,
and what that means for reproducing archived numbers. Validation evidence is
in `docs/VALIDATION.md`.

Baseline: this code is architecturally the n=200 pipeline (the most recent,
most fully fixed of the three), parameterized by sample size.

## D1. No-cross-fit seeding (affects n=1200 archived rows only)

- **Production:** the NoCF path was seeded (`sim_seed*1000 + 11`) at n=200
  and n=5000, but historical n=1200 NoCF runs were **unseeded** — their
  results are run-specific draws of each learner's internal randomness.
  *(Update 2026-07-28: the production n=1200 code is now fixed too —
  the one-line seed was ported into `Naimi_streamlined.v4/analysis/modules/
  nuisance.R` per SEED_AND_DATA_ALIGNMENT.md §9.1 — so future production
  NoCF runs are seeded; every claim below about the ARCHIVED rows stands.)*
- **This code:** seeds the NoCF path at every n.
- **Consequence:** archived n=1200 NoCF rows for stochastic learners
  (lasso, ranger, xgboost, xgboost_naimi, sl_*, hal_discrete) are not
  bit-reproducible by any code, including the original. This code's NoCF
  numbers agree with them only in distribution (Monte-Carlo-level agreement).
  Deterministic learners (parametric, gam, knn) and content-seeded learners
  (mlp, bart) are unaffected and reproduce archived values exactly.
  CF rows are pinned by the fold seed and reproduce exactly, with two
  remaining exceptions: hal_discrete CF (D8) and the n=1200 v3-imported
  cells (fold partitions from the deleted v3 pipeline). A third exception —
  the n=1200 backfill cells, whose sims were mislabeled by one (ISSUES §6)
  — was REMEDIATED IN PRODUCTION 2026-07-05: the archived CSVs were
  relabeled and the unified file rebuilt, so those cells now reproduce
  exactly against the current archive with no shift needed. (The
  VALIDATION.md §2 "EXACT after shifting sim_id by one" note describes the
  pre-remediation archive it was run against.)

## D2. Explicit TMLE gbound at all sample sizes

- **Production:** only the n=200 directory passed `gbound` to `tmle::tmle()`.
  The parents passed nothing, so tmle() applied its internal adaptive floor
  `5/(sqrt(n)·ln n)` to each arm's weight denominator (0.0204 at n=1200,
  0.0083 at n=5000). Because the learner-level truncation [0.025, 0.975] is
  tighter than those floors, this changed **nothing** for the parents'
  default-truncation rows — but at n=200 the floor (0.0667) would have
  overridden the study truncation, which is why the n=200 directory added
  the pass-through.
- **This code:** always passes `gbound = pi_bounds`. Numerically a no-op at
  n=1200/5000 relative to production; it makes the truncation policy explicit
  and identical at every n.
- One knock-on difference: `oracle_notrunc`'s TMLE is re-bounded to
  [0.025, 0.975] here (identical to `oracle`'s TMLE, as in production
  n=200), whereas the parents' `oracle_notrunc` TMLE was floored at the
  weaker package default instead. IPW/G-comp/AIPW are genuinely untruncated
  in both.

## D3. TabPFN cloud track: one DGP, paired data (affects n=1200/n=5000)

- **Production:** at n=1200/n=5000 the TabPFN cloud notebooks generated their
  **own** datasets with numpy RNG (`seed = sim_id`) — same DGP, different
  realizations, so the cloud track was unpaired with the R learners and
  per-sim joins across tracks were invalid. At n=200 the v3 track read
  R-exported CSVs (paired) — the corrected design.
- **This code:** the R stage-1 cache is exported to CSV and the Python step
  reads it at every n. The TabPFN arm is per-sim paired with all R learners.
- **Consequence:** clean-code `tabpfn_v3` per-sim rows at n=1200/5000 will
  NOT match archived cloud-track rows (different realizations); agreement is
  expected at the aggregate level only. At n=200 they match up to D5.

## D4. TabPFN cross-fitting folds

- **Production:** the Python track shuffled its own folds (numpy,
  seed = sim seed) — a partition unrelated to the R learners' folds.
- **This code:** stage 1 exports the shared R fold assignment
  (`sim_seed*1000 + 7`) in the dataset CSV and the Python step uses it, so
  the TabPFN CF arm uses the SAME folds as every R learner.

## D5. TabPFN estimators run in R (TMLE fluctuation unified)

- **Production:** the Python track computed its own estimators. IPW/G-comp/
  AIPW were exact ports of the R math, but its TMLE used a **linear**
  fluctuation on raw Y, not `tmle::tmle()`'s logistic fluctuation on scaled
  Y — a documented cross-track inconsistency.
- **This code:** Python writes only nuisance vectors (raw, untruncated);
  R applies the same truncation and runs the same four estimators used for
  every other learner, including `tmle::tmle()` with explicit gbound.
- **Consequence:** given identical nuisances, IPW/G-comp/AIPW reproduce
  archived n=200 v3 values to float precision; TMLE values are close but not
  identical (different targeting algorithm — the more consistent one is used
  here).
- **Update 2026-07-24 — production now carries this code's TMLE as a named
  learner:** the production n=200 study added `tabpfn_v3_api_tmleR` — the real
  `tmle::tmle()` (logistic fluctuation, explicit `gbound = pi_bounds`)
  recomputed offline on the SAVED tabpfn_api_v3 raw nuisance fits (verified ≡
  production fits at 1.8e-15; `Naimi_v4.n200/analysis/TabPFN_API_v3/
  compute_tmleR_from_raw_fits.R`, archived per-sim values in
  `Naimi_v4.n200/results/tabpfn_api_v3_tmleR/`). **This code's `tabpfn_v3`
  TMLE rows correspond to that learner, not to the archived `tabpfn_api_v3`
  TMLE rows** — same estimator, same gbound convention. The archived
  `tabpfn_api_v3` TMLE (linear fluctuation) remains the production headline;
  the fluctuation-model effect between the two is measured
  (Naimi_streamlined.v4/analysis/TMLE_fluctuation_check/): CF arms
  immaterial (mean |Δψ̂| ≈ 0.06, no coverage flips), NoCF arms shift bias
  (complex 1.53→1.11, simple 0.50→0.09) with coverage still collapsed either
  way.
- **Correction 2026-07-26 — the float-precision claim needs a qualifier this
  entry previously lacked.** Agreement to float precision holds only when
  this code's estimator path is fed the **saved** production nuisance fits
  (the offline route VALIDATION.md §3 actually ran). It does NOT hold for a
  fresh API run: the v3 cloud endpoint drifts server-side over weeks
  (measured 2026-07-24 at n=1200 — fresh re-fits of the May-2026 production
  inputs moved per-sim ATEs by up to 0.16, including 0.034 on seed-free,
  π-free G-computation). "Deterministic at random_state = 0" is true within
  a run window only.
- **Status of the production extension to n=1200/n=5000 (as of 2026-07-26):**
  production is extending this track under two learner names —
  `tabpfn_api_v3_regen` (fresh v3 fits + the four Python estimators; a new
  internally-consistent realization, aggregate-comparable to `tabpfn_api_v3`)
  and `tabpfn_v3_api_tmleR` (R `tmle::tmle()` on those same fresh fits).
  n=1200 completed 2026-07-25 (`Naimi_streamlined.v4/results/
  tabpfn_api_v3_regen_outputs/` + `tabpfn_api_v3_tmleR/`); the n=5000 raw-fit
  regeneration completed its 800-cell API run 2026-07-26. Plan + evidence:
  `~/Documents/Thesis_Aim2/tmleR_extension_n1200_n5000_PLAN.md` (REVISION v2).
  Note for cross-referencing: at n=1200/5000 those production tracks run on
  the numpy datasets (D3), so this code's `tabpfn_v3` rows remain per-sim
  UNPAIRED with them — aggregate comparisons only.

## D6. HAL representative at n=5000

- **Production:** n=5000 could not afford `hal_discrete` (~14 min/fit) and
  ran a deliberately simplified single-fit `hal` (max_degree 2, smoothness 1,
  internal CV = 5-fold) as its primary "HAL".
- **This code:** implements `hal_discrete` (the n=200/n=1200 HAL) at all n
  for protocol consistency.
- **Consequence:** running `hal_discrete` at n=5000 here does not reproduce
  the archived n=5000 `hal` rows — it is a different (richer, much slower)
  model. For exact reproduction of those rows, port the simplified learner
  from `Naimi_streamlined.v4_n5000/analysis/modules/learners/hal.R`.

## D7. Scope reductions (deliberate)

- **Truncation arms:** production n=200 supported runtime-selectable arms
  (default / none / adaptive). The truncation sensitivity analysis was
  concluded in June 2026 (default remains primary; adaptive closed as a
  passed robustness check; `none` never run as production). This code
  implements the primary arm only ([0.025, 0.975]).
- **Learner roster:** core manuscript roster only. Not ported: the tuned
  search learners (ranger_tuned, xgboost_tuned, xgboost_probst), secondary
  HAL variants (hal, hal_d3s1, hal_d3s0, hal_discrete_rb), sl_hal_d3s0,
  sl_tabpfn, local TabPFN (tabpfn, tabpfn_n4), the TabPFN v2.5 cloud track,
  sl_naimi_v2_discrete, the SL+TabPFN hybrids, DoPFN, and the historical
  tabpfn_int.
- **Result schema:** simplified — no `truncation` column (single arm), no
  `runtime_wall` / `nuisance_time_wall_serial_eq` columns, no multi-source
  provenance columns (single pipeline, one canonical output per n).

## D8. hal_discrete cross-fitting uses deterministic parallel RNG streams

- **Production:** the CF fold loop ran under `mclapply` with the default
  `mc.set.seed = TRUE` under Mersenne-Twister, which re-seeds every fold
  worker from time + process ID. Consequence (verified empirically during
  validation): two runs of the same (sim, config) with identical seeding
  produce different hal_discrete CF nuisances (max |Δ pihat| ≈ 0.17 on
  n=200 sim 1) — the archived hal_discrete CF rows are **not reproducible
  by any code, including production itself**. The fits are still valid HAL
  fits; only bit-reproducibility is lost.
- **This code:** draws one seed from the ambient (fold-seed-pinned) stream,
  switches to L'Ecuyer-CMRG so each fold job receives its own deterministic
  stream, and restores the caller's RNG kind/state afterwards. Verified:
  identical results across repeated runs and across `HAL_CF_CORES`
  settings.
- **Consequence:** clean-code hal_discrete CF rows will not match archived
  hal_discrete CF rows (nothing can); they are statistically equivalent
  draws and, going forward, exactly reproducible. hal_discrete NoCF rows
  reproduce archived values exactly (no forking on that path).

## D9. Cross-n protocol facts retained as-is (not deviations, stated for methods)

- CF folds and learner-internal CV folds are **10 at n=200 vs 5 at
  n=1200/5000** (config `folds_by_n`), exactly as in production. This is an
  accepted protocol difference across sample sizes; state it in methods.
- `mlp` uses maxit = 250 (a deliberate production decision, vs SL.nnet's
  500); `bart` uses ndpost = 500 (vs dbarts default 1000). Kept.
- hal9001's internal lambda CV in `hal_discrete` is the package-default
  10-fold at every n (hardcoded upstream), independent of `folds_by_n`.
- Nothing is paired ACROSS sample sizes (different draws per n); all
  learners are paired WITHIN a sample size.

## D10. TabPFN cloud calls get a transient-failure retry wrapper (added 2026-07-15; infrastructure-only, estimates unchanged)

- **Production:** no retry logic anywhere in the Naimi directories' Python
  track — a single transient cloud blip (`RuntimeError: TabPFN is
  inaccessible at the moment ...`, a documented server-side condition with
  observed outages up to ~31 min) failed the affected unit of work.
- **This code:** every `.fit()/.predict()/.predict_proba()` call in
  `python/tabpfn_v3_nuisance.py` goes through `_retry()` — ported verbatim
  from the ACIC production module
  (`ACIC_2016/analysis/modules/python/tabpfn_nuisance_v3.py`, added there
  2026-07-02): transient-only (fail-fast on auth/credit/unknown errors),
  6 tries with deterministic 5→80s capped exponential backoff, re-raise
  after exhaustion so the skip-existing resume stays the backstop. Env
  knobs `TABPFN_RETRY_{TRIES,BASE,CAP}` (defaults 6 / 5s / 120s).
- **Consequence:** none for any estimate — `random_state` is fixed, so a
  retried fit is bit-identical to an unretried one; only whether a call
  survives an outage changes. This is an operational robustness addition,
  not a statistical deviation.

## D11. 2026-08-15 packaging pass (three-script consolidation)

For submission, the modular pipeline (`run.R` + `R/` modules + `R/learners/`)
was consolidated into the three scripts in this directory by verbatim
concatenation of the validated module files. Function bodies are byte-identical
to the validated implementation; the only changes are:

- **Driver/CLI**: one shared header per script replaces `run.R`; stage 2b
  became `02 --track tabpfn`.
- **Roster restricted to the manuscript**: learners that appear in no
  manuscript or supplementary table (`mlp`, default-parameter `xgboost`,
  `sl_naimi_v1`, `oracle_notrunc`) are not included here. They remain in the
  production archives.
- **`hal_n5000` added**: the deliberately simplified single-fit HAL that the
  study ran as its n=5000 "HAL" (D6) is now ported into this codebase
  (verbatim algorithm from `Naimi_streamlined.v4_n5000/analysis/modules/
  learners/hal.R`, including its deterministic per-fold CF seeding
  `sim_seed*1000 + 70 + k`), selected automatically at `--n 5000` via
  `config estimation.hal_by_n`. `estimate_nuisance()` now exposes
  `config$sim_seed` to support this (inert for all other learners —
  re-validated, see below).
- **TabPFN "no aggregation" arm added**: `python/tabpfn_v3_nuisance.py`
  gained `--n_estimators`; a non-default value writes to its own nuisance
  directory and `--track tabpfn` labels those rows `tabpfn_v3_ne<k>`
  (the manuscript's n=1200 sensitivity arm uses ne1).
- Post-hoc n=200 variants (Mitra, SL+TabPFN ensembles; supplementary
  Table S4) are NOT ported — production-pipeline only (see README).

**Re-validation (2026-08-15, n=200 sims 1–2):** datasets and fold exports
byte-identical to the modular pipeline; per-config estimator rows
(estimate/SE/CI/covered) max |Δ| = 0 for parametric, gam, lasso, knn, ranger,
xgboost_naimi, oracle (both scenarios × both CF arms), for the TabPFN track
fed identical nuisance vectors, and for the stage-3 summary metrics
(128 cells). hal_n5000 CF verified deterministic across `HAL_CF_CORES`
settings. Because the modular pipeline was itself validated against the
production archives (docs/VALIDATION.md), these scripts inherit that
validation chain.
