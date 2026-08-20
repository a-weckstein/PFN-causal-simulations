**Pretrained tabular foundation models with prior‑data fitted networks for causal nuisance estimation**
Pipeline for generating simulated datasets, implementing nuisance learners, and building causal estimators.

See [Data_generation.md](Data_generation.md) for overview of the data-generating pipelines for the fully synthetic simulation (1) and the plasmode (semi-synthetic) simulation (2).

# Simulation (1) — Naimi/Kang–Schafer fully synthetic simulation

* Self-contained code for the manuscript's fully synthetic simulation (1)
* The pipeline is three R scripts plus one (optional) Python step, all parameterized by `config.yaml`. Each R script runs from the command line (`Rscript ... --n <k>`) or interactively

| File |  |
|---|---|
| `01_generate_dgp_data.R` | Generates and caches the 200 simulated datasets per sample size (RDS for R track; CSV export incl. shared cross-fitting fold assignment for the Python TabPFN track). |
| `02_run_nuisance_learners.R` | Fits every R nuisance learner per (scenario × learner × cross-fit × sim) cell and saves the nuisance vectors (propensity score + outcome regressions) to `_data_processed/n<k>/nuisance/<learner>/`. 
| `python/tabpfn_v3_nuisance.py` | The only Python step. TabPFN v3 cloud-API nuisance fits (raw, untruncated vectors), saved into the same `nuisance/tabpfn_v3/` layout. |
| `03_run_ate_estimators.R` | Runs all four estimators (IPW, G-computation, AIPW, TMLE) on the saved nuisance vectors of every learner (R and python tracks), writing one CSV per cell to `results/n<k>/per_config/`. |

## How to run
```bash
# 1. Datasets (seed = 1 + sim_id)
Rscript 01_generate_dgp_data.R --n 1200

# 2. R nuisance learners
Rscript 02_run_nuisance_learners.R --n 1200
Rscript 02_run_nuisance_learners.R --n 1200 --learners parametric,ranger --sims 1:20

# 3. TabPFN v3 nuisance fits (optional, uses the TabPFN cloud API, or run on local CPU/GPU)
pip install -r python/requirements.txt        # once: python -c "import tabpfn_client; tabpfn_client.init()"
python python/tabpfn_v3_nuisance.py --n 1200

# 4. ATE estimators over every saved nuisance fit (R learners + tabpfn_v3)
Rscript 03_run_ate_estimators.R --n 1200
```
All scripts accept `--sims a:b` / `--sims a,b,c`; scripts 02 and 03 also accept `--learners`, `--scenarios`, and `--cross_fit` subsets. Per-sim results are written to `results/n<k>/per_config/` 

# Simulation (2) — ACIC 2016 semi-synthetic plasmode simulation

*Forthcoming*

# References

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
- Dorie V, Hill J, Shalit U, Scott M, Cervone D (2019). Automated versus
  do-it-yourself methods for causal inference: Lessons learned from a data
  analysis competition. *Stat Sci* 34(1):43–68. (ACIC 2016.)
