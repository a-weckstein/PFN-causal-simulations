# VALIDATION_RECORD — 2026-08-15 packaging validation (this machine, R 4.5.2)

Comparison target: the modular `naimi_manuscript_code` pipeline (itself
validated EXACT against the production archives — see docs/VALIDATION.md).
Both pipelines were run fresh in a scratch area on identical inputs.

| Check | Result |
|---|---|
| Stage-1 datasets (RDS, field-by-field) + CSV/fold exports, n=200 sims 1–2 | byte-identical |
| Per-config estimator rows (estimate/SE/CI/covered), n=200 sims 1–2, both scenarios × both CF arms: parametric, gam, lasso, knn, ranger, xgboost_naimi, oracle | max abs diff = 0 (28/28 files) |
| TabPFN track (`--track tabpfn`) on identical synthetic nuisance vectors, all 4 estimators | max abs diff = 0 |
| `tabpfn_v3_ne1` variant plumbing (separate nuisance dir → learner label) | correct |
| Stage-3 summary metrics (128 cells, all columns except nuisance-time) | max abs diff = 0 |
| `hal_n5000` no-CF smoke run (n=200 sim 1) | runs; sane estimates |
| `hal_n5000` CF determinism across HAL_CF_CORES=2 vs 4 | max abs diff = 0 |

Only timing columns (`runtime`, `nuisance_time*`) differ between runs, as
expected. Learners not re-run here (bart, sl_*, hal_discrete) are verbatim
byte-copies of modules covered by docs/VALIDATION.md.
