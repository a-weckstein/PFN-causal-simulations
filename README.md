# Simulation (1) — Kang–Schafer/Naimi nuisance-learner study: manuscript code

Self-contained code for the manuscript's fully synthetic simulation (1): a
head-to-head evaluation of nuisance learners combined with four average
treatment effect (ATE) estimators (IPW, G-computation, AIPW, TMLE) on the
Kang & Schafer (2007) data-generating process as adapted by Naimi, Mishler &
Kennedy (2023), at n = 200, 1,200, and 5,000 (200 Monte Carlo replicates per
sample size; simple and complex confounding scenarios; with and without
cross-fitting).

The pipeline is three R scripts plus one Python step, all parameterized by
`config.yaml`:

| File | What it does |
|---|---|
| `01_generate_dgp_data.R` | Generates and caches the 200 simulated datasets per sample size (RDS for the R track; CSV export incl. shared cross-fitting fold assignment for the Python TabPFN track). |
| `02_run_learners_estimators.R` | Fits every nuisance learner and runs all four estimators on the shared nuisance vectors, per (scenario × learner × cross-fit × sim) cell. `--track tabpfn` runs the same estimators over the saved TabPFN nuisance fits. |
| `03_compute_metrics.R` | Combines per-sim results and computes the manuscript metrics: bias (+ Monte-Carlo SE), empirical SE, model SE, SE-calibration ratio, RMSE, and 95% CI coverage (+ MC-SE). |
| `python/tabpfn_v3_nuisance.py` | The only Python step: TabPFN v3 cloud-API nuisance fits (raw, untruncated vectors), read back by script 02. `--n_estimators 1` produces the "no aggregation" sensitivity arm. |

## How to run

```bash
# 1. Datasets (deterministic; seed = 1 + sim_id)
Rscript 01_generate_dgp_data.R --n 1200

# 2. R learners + oracle reference arm (resume-safe; supports subsets)
Rscript 02_run_learners_estimators.R --n 1200
Rscript 02_run_learners_estimators.R --n 1200 --learners parametric,ranger --sims 1:20

# 3. TabPFN v3 track (optional; needs a tabpfn-client login):
pip install -r python/requirements.txt        # once: python -c "import tabpfn_client; tabpfn_client.init()"
python python/tabpfn_v3_nuisance.py --n 1200
python python/tabpfn_v3_nuisance.py --n 1200 --n_estimators 1   # "no aggregation" arm (n=1200 only in the manuscript)
Rscript 02_run_learners_estimators.R --n 1200 --track tabpfn

# 4. Summary metrics
Rscript 03_compute_metrics.R --n 1200
```

Runtime is dominated by the SuperLearner ensembles and HAL;
parametric/gam/lasso/knn cells run in seconds per sim. Clear
`results/n<k>/per_config/` after any change to `config.yaml` or the learner
roster — script 03 aggregates every CSV it finds there.

## Learners (manuscript name ← internal name)

| Manuscript | Internal | One-line spec |
|---|---|---|
| Parametric | `parametric` | logistic PS + single linear `Y ~ A + X` (the only non-stratified outcome model) |
| Random Forest | `ranger` | 500 trees, mtry 2, min.node.size CV over {30, 60} |
| HAL (n=200/1200) | `hal_discrete` | discrete SL over HAL(d2, s0) vs HAL(d2, s1) by lasso CV risk |
| HAL (n=5000) | `hal_n5000` | deliberately simplified single-fit HAL (d2, s1) for runtime; see DEVIATIONS.md D6 |
| TabPFN | `tabpfn_v3` | TabPFN v3 cloud API, out-of-the-box (n_estimators 8, random_state 0) |
| TabPFN (no aggregation) | `tabpfn_v3_ne1` | as above with n_estimators = 1 (n=1200 sensitivity arm) |
| SL Naimi | `sl_naimi_v2` | SuperLearner: RF×2 + XGBoost×2 + GAM(deg 3:8) on main terms + all pairwise interactions |
| SL Balzer | `sl_balzer` | SuperLearner: GLM + stepwise interactions + MARS + mean |
| SL Default | `sl_default` | tmle-package default libraries (GLM + BART + GAM / glmnet) |
| XGBoost | `xgboost_naimi` | 500 rounds, depth 4, eta 0.1, min_child_weight CV over {30, 60} |
| GAM | `gam` | `SL.gam`, deg.gam = 4 |
| BART | `bart` | `dbarts`: probit-BART PS + Gaussian BART outcomes; ntree 200, ndpost 500 |
| LASSO | `lasso` | `cv.glmnet` (α = 1, λ.min) over main effects + all 2-way interactions |
| KNN | `knn` | `caret::knnreg`, k CV-selected from {5,…,50}, standardized |
| Oracle | `oracle` | true nuisance functions (π truncated like every learner) — reference ceiling |

Estimated propensities are truncated to the fixed bounds [0.025, 0.975]
inside every learner (and the same bounds are passed to `tmle::tmle()` as
`gbound`). Cross-fitting is single ("DML2"/pooled): K learner-independent
folds shared by every learner within a simulation (K = 10 at n=200, 5 at
n=1200/5000; learner-internal tuning CV uses the same fold counts). All four
estimators consume the same nuisance vectors, so estimator contrasts are
paired within learner and learner contrasts are paired within simulation.

The post-hoc n=200 variants reported in the supplement (Mitra via
AutoGluon, and the SL-ensemble-with-TabPFN hybrids) were run in
the production n=200 pipeline and are not part of this minimal codebase;
their specifications are in supplementary Table S1 and their implementations
are available from the authors.

## Reproducibility

Every random draw derives from `sim_seed = starting_seed + sim_id`: the
dataset (`set.seed(sim_seed)`), the shared cross-fitting folds
(`sim_seed*1000 + 7`), the full-sample fits (`sim_seed*1000 + 11`), and
hal_n5000's per-fold streams (`sim_seed*1000 + 70 + k`). `bart` additionally
seeds each fit from a checksum of its training data. Any subset of sims can
therefore be re-run in any order and reproduces identical numbers for every
R learner (verified; see docs/VALIDATION.md).

**TabPFN v3 is the one exception.** The cloud API is deterministic at
`random_state = 0` only within a time window: identical requests reproduce
each other during a run, but the served model drifts over weeks. Archived
TabPFN results are reproducible only from their saved nuisance fits, not by
a fresh API call (DEVIATIONS.md D5).

R packages: SuperLearner, tmle (2.x), sandwich, ranger, xgboost, glmnet,
gam, earth, caret, dbarts, hal9001, yaml. Validated under R 4.5.2; see
docs/VALIDATION.md for package versions and the archived-results
reproduction evidence, and DEVIATIONS.md for the documented differences
between this consolidated code and the original production runs.

## References

- Naimi AI, Mishler AE, Kennedy EH (2023). Challenges in obtaining valid
  causal effect estimates with machine learning algorithms.
  *Am J Epidemiol* 192(9):1536–44.
- Kang JDY, Schafer JL (2007). Demystifying double robustness. *Stat Sci*
  22(4):523–39.
- Chernozhukov V et al. (2018). Double/debiased machine learning.
  *Econom J* 21(1):C1–C68. (DML2 pooled cross-fitting.)
- Balzer LB, Westling T (2023). Demystifying statistical inference when
  using machine learning in causal research. *Am J Epidemiol* 192(9):1545–9.
- Hollmann N et al. (2025). Accurate predictions on small data with a
  tabular foundation model. *Nature* 637:319–26. (TabPFN; v3 cloud API.)
