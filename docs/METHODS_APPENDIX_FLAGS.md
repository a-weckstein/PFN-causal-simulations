# METHODS_APPENDIX_FLAGS.md — issues and inconsistencies noticed while writing the technical appendix (2026-07-20)

Findings from the documentation pass that produced
[`METHODS_TECHNICAL_APPENDIX.md`](METHODS_TECHNICAL_APPENDIX.md). Everything
already tracked in [`DEVIATIONS.md`](../DEVIATIONS.md),
[`DEVIATIONS_vs_upstream.md`](../DEVIATIONS_vs_upstream.md), or
[`ISSUES_AND_IMPROVEMENTS.md`](ISSUES_AND_IMPROVEMENTS.md) is only
cross-referenced (Section D); Sections A–C are new observations. None of
these invalidates any statistic; they are labeling, robustness, and
analysis-foot-gun items.

---

## A. Status observations (important for anyone reading this directory)

### A1. The stored `results/` are validation-scale, NOT the study results

`results/n200/summary_metrics_n200.csv` and `figures_n200.pdf` are built from
**2–3 simulations per cell** (verified: `n_sims` ∈ {2, 3} across all 272
n=200 summary cells; all 120 n=1200 cells have `n_sims = 3`), and
`results/n5000/` contains only the parametric learner (4 per-config files, no
summary/figures). These are the residue of the VALIDATION.md runs. The full
200-simulation archives live in the three production directories
(`Naimi_v4.n200`, `Naimi_streamlined.v4`, `Naimi_streamlined.v4_n5000`).

**Risk:** a collaborator (or future you) grabbing `summary_metrics_n<k>.csv`
from this "manuscript code" directory would silently use 3-sim numbers.
**Resolution (2026-07-26):** `results/RESULTS_ARE_VALIDATION_ONLY.md` marker
added and the README "How to run" section now states it. (The stage-4 figure
subtitles do print the sim count, which mitigates but doesn't remove the CSV
risk.)

## B. Code-level observations (minor, behavior-affecting only at the margins)

### B1. Stage 2b's resume check is weaker than stage 2's — **FIXED 2026-07-26**

`stage2_estimate.R` counts a sim complete only if **every estimator row is
present with a non-NA estimate**, and re-queues anything else
([`R/stage2_estimate.R:82–98`](../R/stage2_estimate.R)). Stage 2b used to mark
a sim done if *any* rows existed for it — no completeness or non-NA check, so
a tmle() error in the TabPFN track would have left a permanent NA row that a
re-run never repairs. Stage 2b now applies the same completeness rule and
re-queues incomplete sims
([`R/stage2b_estimate_python_track.R:48–63`](../R/stage2b_estimate_python_track.R)).

### B2. `tabpfn_v3` rows carry `nuisance_time = NA`

The Python script measures per-unit elapsed time but only prints it to
stdout; stage 2b writes `nuisance_time = NA_real_`. So `mean_nuisance_time`
is NaN for every tabpfn_v3 summary cell, and any runtime figure/table built
from this pipeline's output silently lacks the TabPFN arm (on top of the
CPU-vs-wall caveat already in ISSUES §13). If TabPFN timing matters for the
manuscript, persist the Python elapsed time into the nuisance CSVs (an extra
column or sidecar) and carry it through stage 2b — or source TabPFN timing
exclusively from the dedicated timing-harness tracks.

### B3. `mlp` and `bart` share an identical content-seed function

`.mlp_seed()` and `.bart_seed()` are byte-for-byte the same checksum
(serialize → sum of bytes mod 2³¹−1), so both learners fit under the *same
integer seed* for the same (Y, A, X). Harmless here — they consume the seed
in different RNG streams/algorithms — but worth knowing: (i) it is a weak,
permutation-insensitive hash (byte-sum), so distinct datasets can collide
(consequence-free: a collision only means two fits share a seed); (ii) if a
third content-seeded learner is ever added by copy-paste, document that
"content-seeded" does not mean "learner-unique seed".

### B4. `knn`'s internal CV folds are systematic and data-order-dependent

`.knn_best_k()` assigns fold $f = i \bmod V$ in row order. For this study the
rows are iid draws in generation order, so systematic folds are exchangeable
with random ones and the determinism is a feature. But the learner is not
portable as-is: on any dataset with meaningful row order (sorted, clustered,
time-ordered), systematic folds would bias the CV risk. Fine for the
manuscript; add one caveat sentence if the learner file is ever reused
outside this DGP.

### B5. Stage 3 aggregates whatever sits in `per_config/`

`stage3_summarize.R` reads every CSV in `results/n<k>/per_config/` with no
config fingerprint beyond what's in the rows. Stale files from an earlier
roster/config (e.g. a learner later renamed, or sims run under a since-edited
config) would be merged into `all_sim_results` without complaint. The stage-1
cache-validation guard protects the *data*; nothing analogous protects the
*results* directory. Low risk in practice (single-config design), but "clear
`per_config/` after any config change" deserves a line in the README.

## C. Analysis foot-guns (correct code, easy to misuse downstream)

### C1. Oracle rows are duplicated under both cross-fit labels

Stage 2 writes identical oracle rows with `cross_fit = FALSE` and `= TRUE`
(CF is a no-op for known truth — documented in the stage header). Any
downstream "effect of cross-fitting" summary that averages over learners
without excluding `oracle`/`oracle_notrunc` will shrink the CF contrast
toward zero by construction. Same for learner-average tables: the oracle arms
should be a reference overlay, not roster members (stage 4 handles this
correctly; external consumers may not).

### C2. `nuisance_time` is repeated on all four estimator rows

Each estimator row within a (sim × config) carries the *full* shared
nuisance-fit time. Summing `runtime + nuisance_time` across rows
quadruple-counts the nuisance fit; per-cell means (as in stage 3) are fine.
Worth one sentence wherever total-compute numbers are reported.

### C3. IPW SE convention and type

Two stacked conventions, both worth a methods sentence (the appendix now has
one): the sandwich treats estimated weights as fixed (ISSUES §12, usually
conservative), and the variant used is HC0 (`type = "HC"`), the small-sample
*least* conservative HC variant — at n = 200 these two push coverage in
opposite directions. Not a defect (matches upstream); just don't interpret
IPW coverage at n = 200 as purely the weights-as-fixed story.

### C4. `se_ratio` mixes SE constructions across estimators

`mean_se` averages IPW's sandwich SE, AIPW's influence-function SE, and
TMLE's IC SE — each a different analytic construction. Comparing `se_ratio`
*across estimators* is a comparison of different SE estimators, not of one
method's calibration; fine within-estimator across learners. Label
accordingly in any table that shows it.

## D. Known items re-encountered (already documented; no new action)

| Item | Where documented |
|---|---|
| n=1200 historical NoCF runs unseeded → distributional reproduction only | DEVIATIONS D1; ISSUES §1 |
| TMLE gbound explicit at all n; oracle_notrunc TMLE ≡ oracle TMLE | DEVIATIONS D2 |
| TabPFN pairing/folds/TMLE-fluctuation unified vs production | DEVIATIONS D3–D5; ISSUES §3 |
| `hal_discrete` ≠ archived n=5000 `hal` | DEVIATIONS D6; ISSUES §4 |
| hal_discrete CF archived rows irreproducible; fixed via L'Ecuyer streams | DEVIATIONS D8; ISSUES §9 |
| 10 vs 5 folds across n; hal9001 internal CV hardcoded 10-fold | DEVIATIONS D9; ISSUES §5 |
| n=1200 CF backfill sim-labeling off-by-one (remediated) | ISSUES §6 |
| Python-track `seed` column off-by-one in *production* (join on `sim_id`) | ISSUES §7 |
| `sl_naimi_v1/v2` naming vs the paper's "Version 1/2" | DEVIATIONS_vs_upstream C3 |
| Upstream CODE's treatment-assignment complement vs its PAPER | DEVIATIONS_vs_upstream B3 |
| Timing columns not comparable across learners (CPU vs wall) | ISSUES §13 |
| No package-version pinning (renv) | ISSUES §17 |

## E. Documentation cross-checks that PASSED (recorded so they aren't re-litigated)

- README learner one-liners match the code for all 15 learners (grids,
  depths, rounds, libraries, determinism claims) — verified line-by-line.
- `config.yaml` seed comment (sims 1..200 → seeds 2..201) matches
  `folds_and_seeds.R` and `stage1_simulate.R`.
- Lasso design "10 columns at p = 4" matches `model.matrix(~ .^2)` (4 main +
  6 pairwise).
- `sl_default` libraries match the tmle-package defaults
  (g: `SL.glm + tmle.SL.dbarts.k.5 + SL.gam`; Q: `SL.glm + tmle.SL.dbarts2 +
  SL.glmnet`).
- AIPW compact denominator `(2A−1)π̂ + (1−A)` algebraically equals the
  standard A/π̂ − (1−A)/(1−π̂) EIF form.
- tmle's adaptive floor quoted in `estimators.R` (0.0667 at n = 200; 0.0204
  at 1200; 0.0083 at 5000) recomputes correctly from 5/(√n·ln n).
- Stage 2b rows take `seed` from the RDS cache (`sim_seed = sim_id + 1`), so
  the production off-by-one seed label (ISSUES §7) does **not** recur in this
  codebase.
- `covered` uses tmle's own IC-based CI for TMLE and 1.96-Wald elsewhere —
  consistent with the appendix text.
