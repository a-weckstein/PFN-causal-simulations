# DEVIATIONS_vs_upstream.md — this study vs. Naimi, Mishler & Kennedy (2023)

Where this study's Naimi-family pipeline **faithfully reproduces** the original
work of Naimi, Mishler & Kennedy (2023, *Am J Epidemiol* 192(9):1536–44;
arXiv:1711.07137) versus where it **deviates**, and — critically — which
choices **cannot be verified against the public materials at all**.

This file is distinct from `DEVIATIONS.md`, which documents divergences *among
this study's own three production directories*. Here "upstream" means two
external sources:

- **PAPER** — the published article and its stated methods.
- **CODE** — the authors' public R package
  `github.com/amishler/nonparametricDoublyRobust` (single source file
  `R/functions_analysis.R`, 710 lines; line numbers below refer to it).

> **Governing caveat.** The public repo is a **function library**, not a
> runnable study: it contains no driver that sets the coefficients, sample
> sizes, learner libraries, fold counts, or number of simulations. Therefore
> estimator/nuisance/DGP **formulas** are verifiable against CODE, but the
> **run configuration and learner hyperparameters** are verifiable only against
> the PAPER (or not at all). Every claim below is tagged with its evidence
> source.

Verification date: 2026-07-18. Line references to this study cite the
production pipeline `Naimi_v4.n200/` and the consolidated reimplementation
`naimi_manuscript_code/`, which share estimator/DGP logic.

---

## A. Faithful reproductions (verified)

### A1. Four ATE estimators — algebraically identical (CODE)

| Estimator | Upstream (`functions_analysis.R`) | This study | Status |
|---|---|---|---|
| IPW | stabilized Hájek weights → `lm(Y~A, weights)` → HC sandwich SE (L356–363) | `R/estimators.R:29–37` | **identical** |
| AIPW | EIF `((2A−1)(Y−μ))/((2A−1)π+(1−A))+μ1−μ0`, SE=`sd/√n` (L378–385) | `R/estimators.R:44–56` | **identical** |
| G-comp | `mean(μ1−μ0)`, SE=NA (bootstrap optional, off by default) (L236–239) | `R/estimators.R:39–42` | **identical** |
| TMLE | `tmle::tmle(Y,A,X,Q,g1W)` on pooled OOF nuisances (L401–408) | `R/estimators.R:58–68` | identical **except gbound — see B1** |

The IPW weight expression, the AIPW efficient-influence-function, and the
G-computation contrast are byte-for-byte the same expressions.

### A2. Nuisance outcome-model convention (CODE)

Upstream fits the **parametric** outcome model as a **single pooled**
`Y ~ A + X` GLM scored at A=0/1, with **no A×X interaction**
(`estimate_mu_par`, L92–102). It fits the **ML/SuperLearner** outcome model
**arm-stratified** — a separate fit on `A==0` rows and on `A==1` rows
(`estimate_mu_nonpar`, L196–203). This study reproduces exactly that split:
`parametric.R` pooled; every other learner arm-stratified. **The "parametric =
pooled, all-else = stratified" convention is inherited from upstream, not a
choice of this study.**

### A3. Propensity models (CODE)

Parametric PS = logistic `glm(A ~ X)` (`estimate_pi_par`, L76–82); nonparametric
PS = SuperLearner (`estimate_pi_nonpar`, L120). Reproduced.

### A4. Data-generating process — functional form (CODE + PAPER)

- Confounders: 4 iid N(0,1). Kang–Schafer transforms for the misspecified
  ("complex") scenario are **identical**: Z1=exp(C1/2), Z2=C2/(1+e^{C1})+10,
  Z3=(C1·C3/25+0.6)³, Z4=(C2+C4+20)² (CODE L34–37; this study
  `R/dgp.R:32–35`).
- Outcome linear in C, **homogeneous** additive effect = 6, error SD = 6
  (CODE L64–65; this study β_y=[120,3,3,3,3], `true_ate=6`, `error_sd=6`).

### A5. Treatment/propensity model — matches the PAPER (PAPER)

The paper specifies **P(X=1|C) = expit{−1 + log(1.75)(C1+C2+C3+C4)}**. This
study implements exactly that: β_a = [−1, log1.75, log1.75, log1.75, log1.75],
`A <- rbinom(n, 1, plogis(cbind(1,C) %*% beta_a))` (`R/dgp.R:39–45`), giving
**~31% exposed** as reported. **Faithful to the published DGP.** (See B3 for a
CODE-vs-PAPER inconsistency on this point that does *not* affect this study.)

### A6. Inner SuperLearner cross-validation folds (PAPER)

The paper states the learner-weight CV used **K = 10, 5, 5 for
n = 200, 1,200, 5,000**. This study's `cv_folds = 10/5/5` reproduces this
exactly. (The repo's `cv_folds=5` default is only a function default, not the
paper's run value.)

### A7. SuperLearner ensemble hyperparameters (PAPER)

Every algorithm-level spec matches the paper (the library is defined in
`R/learners/helpers_superlearner.R:91–112`, `create_naimi_library()`;
`sl_naimi_v1.R` is only the registering shim):

| Algorithm | Paper | This study | Status |
|---|---|---|---|
| Random forest | 500 trees, subspace 2, min node {30,60} | `num.trees=500, mtry=2, min.node.size=c(30,60)` | **match** |
| XGBoost | 500 trees, depth 4, shrinkage 0.1, min node {30,60} | `ntrees=500, max_depth=4, shrinkage=0.1, minobspernode=c(30,60)` | **match** |
| GAM | smoothing splines, df 3–8 | `SL.gam, deg.gam=3:8` | **match** |

See **C3** for the ensemble-composition labeling note (the hyperparameters are
faithful; the "v1/v2" *names* do not follow the paper's ordering).

### A8. Propensity truncation value (CODE)

Estimated propensities are truncated to **[0.025, 0.975]**. The paper is
**silent** on truncation, but this is the default of upstream's `bound_pi`
(L219), applied to every π̂ in `simulate_inference` (L540, 543, 557, 558). So
the bound value is faithful to CODE; it is simply undocumented in the article.

### A9. Cross-fitting aggregation — DML2 pooled (CODE)

Upstream fits nuisances out-of-fold (fit on K−1 folds, predict on the held-out
fold; π̂ and μ̂ share one fold vector, L536) and solves each estimator **once
on the pooled out-of-fold vectors** — including a **single pooled TMLE
targeting step** (`est_tmle` calls `tmle()` once on the full-length Q/g1W,
L401–408). This is DML2, and this study matches it. (See B4 on the paper's
"CV-TMLE" wording.)

---

## B. Deviations (declare in the manuscript)

### B1. TMLE `gbound` at n=200 — justified deviation (vs CODE)

Upstream calls `tmle(Y, A, X, Q, g1W = pihat)` with **no `gbound`** (L403), so
`tmle::tmle()` re-bounds g at its internal adaptive floor 5/(√n·ln n) applied
to each arm's weight denominator. Numerically:

- n=1,200 / 5,000: floor = 0.0204 / 0.0083 < 0.025 → **no-op**; TMLE and the
  other three estimators all use [0.025, 0.975]. Faithful, nothing to flag.
- **n=200: floor = 0.0667 > 0.025** → in upstream code, TMLE alone would use
  effective **[0.0667, 0.9333]** while IPW/AIPW/G-comp use [0.025, 0.975].

This study passes `gbound = pi_bounds` explicitly (`R/estimators.R:59–64`), so
**all four estimators share one truncation policy at every sample size.**
This is a deviation from a literal port of upstream code, but it **removes a
latent internal inconsistency** in that code (one estimator truncated
differently from the others, at one sample size) — not a matter of preference.

> *Suggested text:* "At n=200 we pass `tmle()` an explicit `gbound` equal to the
> study truncation [0.025, 0.975]. The public code does not, so `tmle()`'s
> internal floor (≈0.067 at n=200) would truncate the TMLE propensity denominator
> more aggressively than the [0.025, 0.975] used by IPW, AIPW, and
> G-computation. We enforce one truncation policy across all estimators and
> sample sizes; this is a no-op at n=1,200 and 5,000."

### B2. Reproduction is distributional, not per-simulation (CODE)

Even where formulas are identical, this study does **not** reproduce upstream's
exact per-sim numbers: upstream seeds `set.seed(counter)` once per sim and draws
covariates via `rmvnorm`; this study seeds per stage and draws via `rnorm`,
and the treatment draw order differs (B3). Same distributions, different
realized datasets. **State once that reproduction is distributional.**

### B3. Treatment-assignment: upstream CODE contradicts its own PAPER (footnote)

Upstream code assigns `A <- 1 - rbinom(n, 1, expit(Cβ_a))` (L63) — a
**complement**. Under the paper's coefficients that realizes
P(A=1|C) = expit{**+1 − log(1.75)ΣC**} ≈ **69% exposed**, the mirror image of
the paper's stated ~31%. The code matches the paper only if the (unpublished)
driver passed sign-flipped coefficients. **This study follows the PAPER (A5),
which is the source of truth**; note the code's complement as a CODE-vs-PAPER
inconsistency, not a deviation on this study's side.

---

## C. Thesis choices — not verifiable against upstream (label as choices)

### C1. Outer cross-fit fold count `n_splits` = 10/5/5

The paper describes the sample-splitting procedure but **reports no specific
cross-fit K** (its "K folds… predict in remaining… switched" wording hints at
K=2 but is generic). Upstream code does **not fix one either**: `nsplits` is a
function argument with **default 1 = no cross-fitting** (`if (nsplits==1)
train<-test`); when >1 it does generic K-fold cross-fitting. This study's
`n_splits = 10/5/5` (mirroring the inner SL-CV folds) is therefore a
**defensible study choice, not a reproduction** — describe it as such. If the
paper states a cross-fit K elsewhere than the sentence reviewed here, reconcile
against that.

### C2. Extension learners beyond the paper

The paper hints a wider range of SL ensembles was implemented than reported.
This study's additional learners — TabPFN, DoPFN, HAL variants, standalone
BART/MLP/kNN/LASSO/GAM, and the XGBoost variants (Mitra exists only as a
completed n=200 feasibility probe, `Naimi_v4.n200/analysis/Mitra_local/`,
not a study learner) — are **novel extensions**,
consistent-in-spirit with that hint but **not reproductions**. Keep them
clearly separated from the reproduction learners (parametric GLM; the RF+XGB+GAM
SL ensembles) in any faithfulness claim.

### C3. SL ensemble composition — accurate description (terminology only)

The paper's ensembles are: **Version 1** = RF + XGB (no GAM); **Version 2** =
RF + XGB + GAM; and **Version 2 + 2-way interactions**. This study's naming axis
is *interactions vs. not*, with GAM present in both, so the labels are offset:

| This study | Composition | Corresponds to paper's |
|---|---|---|
| `sl_naimi_v1` | RF + XGB + GAM, main effects | **Version 2** |
| `sl_naimi_v2` | RF + XGB + GAM + 2-way interactions | **Version 2 + interactions** |
| *(not implemented)* | RF + XGB only | Version 1 |

**Both study ensembles map to ensembles the paper actually used**, so the
science is fine; the requirement is only that they be **described by
composition** (RF+XGB+GAM ± interactions), *not* as the paper's "Version 1."
NB: `Naimi_v4.n200/_reference/learner_catalog.md:65` currently mislabels
`sl_naimi_v1` as "Naimi et al. 2023, Version 1" — correct this wherever it
feeds the manuscript. The GAM-free Version 1 was deliberately not implemented
(study decision, 2026-07-18).

---

## D. Bugs / quirks observed in the upstream repo (context, not this study's)

- **`est_gcomp_nonparametric` references `A` before it is defined** (L314 uses
  `length(A)` in the `split_inds` fallback; `A` is assigned at L321). Latent —
  survives only because callers always pass `split_inds`.
- **Diagnostics typo** (L570): the observed-arm prediction column is written
  `muhat_F = muhat_F$mu1` instead of `…$mu`. Affects only the saved predictions
  table, not any estimate.
- **Treatment complement** (`1 - rbinom`, L63) — see B3.

None of these affect the estimator outputs; the complement (B3) is the only one
with study-design relevance.

---

## Bottom line

Faithfully reproduced (verified): the four estimators; the pooled-parametric /
arm-stratified-ML nuisance convention; the DGP functional form and the paper's
propensity model; the inner SL-CV folds (10/5/5); the [0.025, 0.975] truncation
value; DML2 pooled cross-fitting including single-targeting TMLE; and all SL
ensemble hyperparameters. Declared deviations: the n=200 TMLE `gbound`
(justified — removes an upstream inconsistency) and distributional-not-per-sim
reproduction. Study choices unverifiable upstream: the outer cross-fit fold
count and the extension learners. Description fix: name the SL ensembles by
composition, not by the paper's Version 1/2 ordering.
