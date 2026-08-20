# Data generation

Overview of the data-generating pipelines for the fully synthetic simulation (1) and the plasmode (semi-synthetic) simulation (2).

* Aligns simulated datasets and cross-fitting fold assignments across learners by caching each per-seed replicate dataset: every learner (R and Python) is evaluated on identical realizations with identical folds, so estimator contrasts are paired within learner and learner contrasts are paired within simulation.

# Simulation (1) — Kang–Schafer/Naimi fully synthetic simulation

* DGP from Naimi, Mishler & Kennedy (2023), adapted from Kang & Schafer (2007): 4 iid standard-normal confounders C1–C4, binary treatment (logistic in C), continuous outcome (linear in C, homogeneous effect), true ATE = 6.
* The "complex" scenario hands the analyst the Kang–Schafer transforms Z = h(C) instead of C, so no learner sees the correctly specified covariates:
  * Z1 = exp(C1/2)
  * Z2 = C2 / (1 + exp(C1)) + 10
  * Z3 = (C1·C3/25 + 0.6)³
  * Z4 = (C2 + C4 + 20)²
* 200 replicates per sample size (n = 200, 1200, 5000), two scenarios (simple/complex). Every random draw derives from `sim_seed = starting_seed + sim_id`, so any subset of sims can be regenerated in any order and reproduces identical datasets.

| File |  |
|---|---|
| `01_generate_dgp_data.R` | Generates and caches the 200 simulated datasets per sample size. Writes per sim: `_data_inputs/n<k>/sim_XXXX.rds` (full dataset incl. ground truth, for the R track) and `_data_processed/n<k>/sim_XXXX.csv` (covariates/treatment/outcomej + the shared cross-fitting fold assignment `fold_id`, for the Python TabPFN track). The CSV is an exact export of the RDS, so the Python track is evaluated on byte-identical data with the same folds as every R learner. Idempotent: existing sims are validated, not rewritten. |

## How to run

```bash
# Datasets (seed = 1 + sim_id); one call per sample size
Rscript 01_generate_dgp_data.R --n 1200
Rscript 01_generate_dgp_data.R --n 200 --sims 1:20   # subsets supported
```

# Simulation (2) — ACIC 2016 semi-synthetic plasmode simulation

*Forthcoming — the ACIC 2016 data-generating pipeline is not yet included in this repok
# References

- Naimi AI, Mishler AE, Kennedy EH (2023). Challenges in obtaining valid
  causal effect estimates with machine learning algorithms.
  *Am J Epidemiol* 192(9):1536–44.
- Kang JDY, Schafer JL (2007). Demystifying double robustness. *Stat Sci*
  22(4):523–39.
- Dorie V, Hill J, Shalit U, Scott M, Cervone D (2019). Automated versus
  do-it-yourself methods for causal inference: Lessons learned from a data
  analysis competition. *Stat Sci* 34(1):43–68. (ACIC 2016.)
