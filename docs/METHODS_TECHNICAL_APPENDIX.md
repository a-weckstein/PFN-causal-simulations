# Technical Appendix — Nuisance-learner simulation study on the Kang–Schafer/Naimi data-generating process (n = 200, 1,200, 5,000)

*Methods-section / web-appendix documentation of the implementation in
`naimi_manuscript_code/`. Written 2026-07-20 against the code as it stands;
every claim below is traceable to a file and line range in this directory.
Companion documents: [`DEVIATIONS.md`](../DEVIATIONS.md) (differences from the
three production directories that generated the archived study numbers),
[`DEVIATIONS_vs_upstream.md`](../DEVIATIONS_vs_upstream.md) (faithfulness to
Naimi, Mishler & Kennedy 2023), [`docs/VALIDATION.md`](VALIDATION.md)
(bit-level reproduction evidence), and
[`docs/ISSUES_AND_IMPROVEMENTS.md`](ISSUES_AND_IMPROVEMENTS.md) (issues found
in the production implementations).*

*Equations are written in plain Unicode text (no LaTeX), so they can be
copied directly into a Word document — pasted as-is, or into Insert →
Equation, whose linear format accepts this notation.*

---

## 1. Study design overview

The study is a fully factorial Monte-Carlo evaluation of machine-learning
nuisance estimation for the average treatment effect (ATE), patterned on
Naimi, Mishler & Kennedy (2023) and the Kang & Schafer (2007) benchmark. The
factors are:

| Factor | Levels | Notes |
|---|---|---|
| Sample size n | 200; 1,200; 5,000 | separate, independently drawn dataset sets per n |
| Simulated datasets | 200 per sample size | seeds 2, …, 201 (Section 10) |
| Covariate scenario | `simple`; `complex` | analyst sees the confounders vs. their Kang–Schafer transforms |
| Nuisance learner | 15 fitted learners + 2 oracle reference arms | Section 7 |
| Cross-fitting | none; K-fold (K = 10 at n = 200, K = 5 otherwise) | DML2 pooled solve (Section 5.3) |
| Estimator | IPW, g-computation, AIPW, TMLE | all four run on the *same* nuisance vectors |

Every learner is evaluated on the **same 200 datasets** within a sample size,
and all cross-fit learners share the **same fold partition** within a dataset.
Consequently, learner contrasts are paired within simulation, cross-fitting
contrasts are paired within (learner, simulation), and estimator contrasts are
paired within (learner, cross-fitting, simulation) — an estimator contrast is
exactly a contrast of estimation strategies applied to identical inputs.
Nothing is paired *across* sample sizes (each n has its own independent
draws).

Pipeline entry point: [`run.R`](../run.R); all study parameters:
[`config.yaml`](../config.yaml). Stage 1 simulates and caches data; stage 2
fits the R learners and runs the estimators; stage 2b runs the estimators on
the Python-generated TabPFN v3 nuisances; stage 3 aggregates; stage 4 draws
standard figures.

## 2. Notation

- X = (X₁, X₂, X₃, X₄): the covariates supplied to a learner in a given
  scenario. In the `simple` scenario X is the confounder vector itself
  (named `C1`–`C4` in code); in the `complex` scenario X is the transformed
  vector Z = h(C) defined in Section 3.2 (code `Z1`–`Z4`).
- A ∈ {0, 1}: binary treatment (code `A`).
- Y: continuous outcome (code `Y`); Y(a) the potential outcome under
  treatment a.
- π(X) = P(A = 1 | X): the propensity score; π̂ its estimate (code `pihat`).
- μₐ(X) = E[Y | A = a, X], a ∈ {0, 1}: the arm-specific outcome
  regressions; μ̂₀, μ̂₁ their estimates (code `mu0hat`, `mu1hat`).
- ψ = E[Y(1) − Y(0)]: the estimand (ATE); true value ψ₀ = 6.
- A superscript star (π∗, μₐ∗) marks the *true* nuisance functions of the
  data-generating process.
- n: sample size; S = 200: number of simulations; K: number of
  cross-fitting folds; V: number of learner-internal cross-validation folds
  (here V = K by design; Section 5.3).
- expit(u) = 1/(1 + e^(−u)); logit(p) = log( p/(1 − p) ).

## 3. Data-generating process

Implementation: [`R/dgp.R`](../R/dgp.R); parameters in
[`config.yaml`](../config.yaml) (`dgp:` block). The DGP is Kang & Schafer
(2007) as adapted by Naimi et al. (2023).

### 3.1 Confounders, treatment, outcome

For each simulated dataset of size n, draw four independent standard-normal
confounders

Cⱼ ~ iid N(0, 1),  j = 1, …, 4.

Treatment is assigned by a logistic model in the raw confounders,

π∗(C) = P(A = 1 | C) = expit{ −1 + log(1.75)·(C₁ + C₂ + C₃ + C₄) },

A | C ~ Bernoulli( π∗(C) ),

with log(1.75) ≈ 0.5596, giving a marginal treated fraction of ≈ 31%. This
matches the propensity model as *published* by Naimi et al.; the public
upstream code contains a sign-flip inconsistency with its own paper on this
point that does not affect this study (see
[`DEVIATIONS_vs_upstream.md`](../DEVIATIONS_vs_upstream.md) §B3).

Potential outcomes are linear in the confounders with a **homogeneous,
additive** treatment effect and a **shared** error draw across arms:

Y(a) = 120 + 3·(C₁ + C₂ + C₃ + C₄) + 6·a + ε,  ε ~ N(0, 6²),

and the observed outcome is Y = A·Y(1) + (1 − A)·Y(0). Because the same ε
enters both arms, Y(1) − Y(0) = 6 for every unit: the individual-level
effect, the conditional ATE at every x, and the marginal ATE all equal
ψ₀ = 6. The DGP also stores the true unit-level nuisances π∗(C),

μ₀∗(C) = 120 + 3·(C₁ + C₂ + C₃ + C₄)  and  μ₁∗(C) = μ₀∗(C) + 6,

for the oracle arms (Section 8).

The random-draw order (confounders → treatment → outcome errors) is fixed in
`simulate_data()` and must not be reordered: it defines the exact datasets on
which every archived result was computed.

### 3.2 The two covariate scenarios

Each learner is run under two covariate regimes on the *same* realized data:

- **`simple`** — the analyst observes X = (C₁, …, C₄). Correct parametric
  specification is possible (the true π∗ is logistic-linear and the true
  μₐ∗ linear in these covariates).
- **`complex`** — the analyst observes only the Kang–Schafer transforms
  X = Z = h(C):

  Z₁ = exp(C₁/2)

  Z₂ = C₂ / (1 + exp(C₁)) + 10

  Z₃ = (C₁·C₃/25 + 0.6)³

  Z₄ = (C₂ + C₄ + 20)²

The map h is one-to-one on a set of probability essentially 1 (given Z₁,
C₁ is recovered exactly; then C₂ from Z₂, C₃ from Z₃ when C₁ ≠ 0, which
holds almost surely; and C₄ from Z₄ up to the sign of C₂ + C₄ + 20, which is
positive except on an event of probability ≈ 10⁻⁴⁵). Conditioning on Z is
therefore equivalent to conditioning on C: **conditional exchangeability
given the observed covariates holds in both scenarios**, and the `complex`
scenario is purely a *functional-form* challenge — the true nuisances are
awkward, non-additive functions of Z that no parametric default captures —
not a violation of identification.

## 4. Estimand and identification

The estimand is the marginal average treatment effect

ψ₀ = E[Y(1) − Y(0)] = 6.

Identification conditions hold by construction: (i) *consistency* — Y =
Y(A) by the composition of the DGP; (ii) *conditional exchangeability* —
(Y(0), Y(1)) ⊥ A | C, since treatment depends only on C, and equivalently
given Z (Section 3.2); (iii) *positivity* — π∗(C) =
expit( −1 + 0.5596·(C₁ + C₂ + C₃ + C₄) ) with C₁ + C₂ + C₃ + C₄ ~ N(0, 4),
so true propensities concentrate well inside (0, 1) (roughly 0.04–0.78 for
±2 SD of the linear predictor), though tail draws can approach the boundary —
the practical-positivity stress that motivates the truncation policy of
Section 5.1. Under (i)–(iii),

ψ₀ = E[ μ₁(X) − μ₀(X) ] = E[ A·Y/π(X) − (1 − A)·Y/(1 − π(X)) ],

and the four estimators of Section 6 are sample-analogue strategies for these
representations.

## 5. Nuisance estimation protocol

Dispatcher: [`R/nuisance.R`](../R/nuisance.R). Every learner exposes two
functions with a common contract: a full-sample fit and a cross-fit variant,
each returning the triple (π̂, μ̂₀, μ̂₁) as n-vectors of unit-level
predictions.

### 5.1 Propensity truncation

Every learner truncates its estimated propensity to the study bounds before
returning:

π̂(Xᵢ) ← min{ max( π̂(Xᵢ), 0.025 ), 0.975 }.

The bounds `[0.025, 0.975]` are set once in `config.yaml`
(`estimation: pi_bounds`) and are the *single* truncation policy for all four
estimators at all sample sizes: IPW/AIPW consume the truncated π̂ directly,
and TMLE receives the same bounds explicitly as its `gbound` argument
(Section 6.4), which prevents `tmle::tmle()` from substituting its own
adaptive floor 5/(√n·ln n) (that floor, 0.0667 at n = 200, is *wider* than
the study bounds and would silently re-truncate). The value 0.025 is
inherited from the upstream Naimi code's `bound_pi` default
([`DEVIATIONS_vs_upstream.md`](../DEVIATIONS_vs_upstream.md) §A8). The
TabPFN track writes *raw* propensities and the identical truncation is
applied in R before estimation
([`R/stage2b_estimate_python_track.R:72`](../R/stage2b_estimate_python_track.R)).

### 5.2 Outcome-model convention: arm-stratified, with one exception

Every machine-learning learner fits the outcome regression **stratified by
arm**: μ̂₀ is fit on the rows with Aᵢ = 0 only and μ̂₁ on the rows with
Aᵢ = 1 only, each then predicted for **all** n units. The single exception
is the `parametric` learner, which fits one pooled Gaussian regression
Y ~ A + X (main effects, *no* A × X interactions) and scores it at A = 0 and
A = 1 — classical parametric g-computation. This "pooled parametric,
stratified ML" split reproduces the upstream Naimi code exactly
([`DEVIATIONS_vs_upstream.md`](../DEVIATIONS_vs_upstream.md) §A2). Note that
under the DGP's homogeneous effect the pooled main-effects model is correctly
specified in the `simple` scenario.

### 5.3 Cross-fitting (DML2, pooled)

Each (scenario × learner) cell is run twice: without cross-fitting (fit on
all n rows, predict the same rows) and with K-fold cross-fitting:

1. A **balanced random partition** of the n units into K folds is drawn once
   per simulated dataset
   (`generate_folds()`, [`R/folds_and_seeds.R`](../R/folds_and_seeds.R)) from
   a learner-independent seed, so **every learner — including the Python
   TabPFN track — uses the identical partition** within a simulation.
2. For each fold k, the learner's *entire* pipeline (including all internal
   hyperparameter tuning; Section 7) is refit on the training rows (all rows
   not in fold k) and predictions are made for the held-out rows of fold k.
   Each unit i thus receives out-of-fold nuisance predictions
   π̂⁽⁻ᵏ⁾(Xᵢ), μ̂₀⁽⁻ᵏ⁾(Xᵢ), μ̂₁⁽⁻ᵏ⁾(Xᵢ) from the one fit that excluded
   its fold k.
3. The full-length out-of-fold vectors are pooled and **each estimator is
   solved once** on them — no per-fold ATE averaging, and a single TMLE
   targeting step on the pooled vectors. This is the "DML2" convention of
   Chernozhukov et al. (2018) and matches the upstream Naimi code's pooled
   solve ([`DEVIATIONS_vs_upstream.md`](../DEVIATIONS_vs_upstream.md) §A9).

Fold counts are K = 10 at n = 200 and K = 5 at n = 1,200 and 5,000
(`config.yaml: folds_by_n`). The larger K at n = 200 keeps CF training sets
at 180 rows (≈ 56 treated on average) rather than 160; the paper's inner
SuperLearner cross-validation used the same 10/5/5 schedule. The same per-n
value drives **both** the outer cross-fitting folds and each learner's
internal CV folds (V = K): a deliberate single-knob design, stated here
because it means internal tuning granularity also changes across sample
sizes (one exception: `hal9001`'s internal lasso CV is hard-coded 10-fold at
every n; Section 7.10).

There is deliberately no generic cross-fitting fallback in the dispatcher: a
learner without an explicit CF implementation fails rather than silently
recycling in-sample predictions.

## 6. Estimators

Implementation: [`R/estimators.R`](../R/estimators.R). All four estimators
consume the same (π̂, μ̂₀, μ̂₁) produced by one learner. Unless noted, 95%
confidence intervals are Wald intervals ψ̂ ± 1.96·SE, and "coverage" refers
to the indicator 1{ lower ≤ 6 ≤ upper }.

### 6.1 Inverse-probability weighting (IPW; Hájek, stabilized)

Stabilized weights

wᵢ = Aᵢ · Ā/π̂(Xᵢ) + (1 − Aᵢ) · (1 − Ā)/(1 − π̂(Xᵢ)),  where Ā = (1/n)·Σᵢ Aᵢ,

are used in a weighted least-squares regression of Y on (1, A); the estimate
is the coefficient on A. With a single binary regressor this is algebraically
the Hájek (ratio-normalized) contrast

ψ̂ = [ Σᵢ wᵢ·Aᵢ·Yᵢ ] / [ Σᵢ wᵢ·Aᵢ ] − [ Σᵢ wᵢ·(1 − Aᵢ)·Yᵢ ] / [ Σᵢ wᵢ·(1 − Aᵢ) ].

The standard error is the heteroskedasticity-consistent sandwich estimator of
the WLS slope (`sandwich::vcovHC`, `type = "HC"` = HC0), **treating the
estimated weights as fixed** — no correction for the estimation of π̂. This
matches the upstream Naimi implementation; for the ATE with estimated
(rather than known) weights this convention is generally conservative, which
should be borne in mind when interpreting IPW coverage above the nominal
level.

### 6.2 G-computation (plug-in)

ψ̂ = (1/n) · Σᵢ { μ̂₁(Xᵢ) − μ̂₀(Xᵢ) }.

**No standard error is computed, by design** (upstream's optional bootstrap
is off by default); g-computation rows carry `se = NA` and are excluded from
all coverage summaries. For the pooled `parametric` learner,
μ̂₁(x) − μ̂₀(x) is constant and ψ̂ equals the fitted coefficient on A.

### 6.3 Augmented IPW (AIPW; one-step efficient estimator)

The estimated efficient influence function is evaluated per unit,

φ̂ᵢ = (2Aᵢ − 1)·( Yᵢ − μ̂(Aᵢ, Xᵢ) ) / [ (2Aᵢ − 1)·π̂(Xᵢ) + (1 − Aᵢ) ] + μ̂₁(Xᵢ) − μ̂₀(Xᵢ),

where μ̂(Aᵢ, Xᵢ) = Aᵢ·μ̂₁(Xᵢ) + (1 − Aᵢ)·μ̂₀(Xᵢ) is the prediction for
the arm actually received. The compact denominator equals π̂(Xᵢ) when
Aᵢ = 1 and 1 − π̂(Xᵢ) when Aᵢ = 0, so this is the familiar form

φ̂ᵢ = (Aᵢ/π̂(Xᵢ))·( Yᵢ − μ̂₁(Xᵢ) ) − ( (1 − Aᵢ)/(1 − π̂(Xᵢ)) )·( Yᵢ − μ̂₀(Xᵢ) ) + μ̂₁(Xᵢ) − μ̂₀(Xᵢ).

Then

ψ̂ = (1/n) · Σᵢ φ̂ᵢ,  SE = sd(φ̂) / √n,

with sd(·) the sample standard deviation (n − 1 denominator).

**Equivalent per-arm presentation.** Regrouping the same terms by treatment
arm, the estimator can be written as a difference of two augmented
arm-specific means, ψ̂ = ψ̂₁ − ψ̂₀, with

ψ̂₁ = (1/n) · Σᵢ [ μ̂₁(Xᵢ) + (Aᵢ/π̂(Xᵢ))·( Yᵢ − μ̂₁(Xᵢ) ) ]

ψ̂₀ = (1/n) · Σᵢ [ μ̂₀(Xᵢ) + ( (1 − Aᵢ)/(1 − π̂(Xᵢ)) )·( Yᵢ − μ̂₀(Xᵢ) ) ].

The two presentations are algebraically identical, not merely
asymptotically equivalent: for every unit i the difference of the two
bracketed summands equals φ̂ᵢ, so they produce the same estimate to the
last digit. The per-arm form makes visible that each potential-outcome mean
E[Y(a)] is itself estimated doubly robustly; the φ̂ᵢ form is the one used
for inference — the standard error is always sd(φ̂)/√n, the SD of the
unit-level *differences*, not something computed from the two arm means
separately. This is the
standard one-step/EIF estimator: doubly robust, and asymptotically efficient
when both nuisances converge fast enough (e.g., both faster than n^(−1/4)),
with cross-fitting removing the Donsker/complexity conditions on the
learners.

### 6.4 Targeted maximum likelihood estimation (TMLE)

TMLE is computed by `tmle::tmle()` (tmle 2.x) with the learner's nuisances
supplied as initial estimates — `Q = (μ̂₀, μ̂₁)` and `g1W = π̂` — so the
package performs **only the targeting step**, never its own nuisance
fitting. The algorithm, as run here:

1. **Outcome scaling.** Y is mapped to [0, 1] by Ỹ = (Y − a)/(b − a) with
   (a, b) the observed range; the initial μ̂ₐ are mapped the same way to μ̃ₐ
   (Gruber & van der Laan 2010).
2. **Propensity bounding.** π̂ is bounded at the explicitly supplied
   `gbound = (0.025, 0.975)` — the study truncation, a no-op for fitted
   learners whose π̂ is already truncated. Passing `gbound` overrides the
   package's internal adaptive floor 5/(√n·ln n), which would otherwise
   re-truncate more aggressively at n = 200 (0.0667) and make TMLE's
   truncation policy differ from the other estimators'
   ([`DEVIATIONS.md`](../DEVIATIONS.md) D2;
   [`DEVIATIONS_vs_upstream.md`](../DEVIATIONS_vs_upstream.md) B1).
3. **Logistic fluctuation.** With clever covariates

   H₁(A, X) = A / π̂(X)  and  H₀(A, X) = −(1 − A) / (1 − π̂(X)),

   a one-step logistic fluctuation submodel is fit by maximum likelihood on
   the scaled outcome with logit μ̃(A, X) as offset (tmle-package default:
   one fluctuation coefficient per arm), yielding targeted arm-specific
   predictions on the scaled scale.
4. **Plug-in and inference.** Back-transformed to the original scale as
   μ̂₀†, μ̂₁†, the estimate is

   ψ̂ = (1/n) · Σᵢ { μ̂₁†(Xᵢ) − μ̂₀†(Xᵢ) },

   with variance from the estimated efficient influence curve

   ICᵢ = [ Aᵢ/π̂(Xᵢ) − (1 − Aᵢ)/(1 − π̂(Xᵢ)) ]·( Yᵢ − μ̂†(Aᵢ, Xᵢ) ) + μ̂₁†(Xᵢ) − μ̂₀†(Xᵢ) − ψ̂,

   estimated variance = sd²(IC)/n, and the package's own IC-based 95% Wald
   CI is reported.

Under cross-fitting the single targeting step runs once on the pooled
out-of-fold nuisance vectors (DML2; Section 5.3) — i.e., this is cross-fit
TMLE with pooled targeting, not fold-wise CV-TMLE.

A related validity check lives outside this codebase: a validated numerical
replica of `tmle::tmle()` showed the linear-vs-logistic fluctuation choice is
negligible on realistic nuisances for this DGP (mean |Δ| ≈ 0.025, no CI
flips over 200 sims) but can diverge under adversarial nuisances — relevant
because the production Python TabPFN track used a linear fluctuation, an
inconsistency this codebase removes by routing every learner through
`tmle::tmle()` ([`DEVIATIONS.md`](../DEVIATIONS.md) D5).

### 6.5 Error handling

An estimator error yields an `NA` row (estimate, SE, CI all `NA`) rather than
a crashed run; a nuisance-fit error yields `NA` nuisance vectors and hence
four `NA` rows. Failed simulations are re-queued on resume and reported (not
silently dropped) by the aggregation stage (Section 9).

## 7. Nuisance learners

One file per learner in [`R/learners/`](../R/learners/). Common contract:
binomial-type fit of A on X for the propensity; Gaussian-type,
arm-stratified fits for the outcomes (except `parametric`, Section 5.2); the
truncation of Section 5.1 applied before returning. Under cross-fitting,
**all** internal tuning described below (CV over λ, k, node sizes, boosting
rounds, SuperLearner weights) is repeated from scratch within each training
fold. V denotes the learner-internal CV fold count (V = K = 10/5/5 by sample
size).

Roster summary (RNG column: how reproducibility is achieved — see Section 10):

| # | Learner | Model class | Key tuning | RNG |
|---|---|---|---|---|
| 1 | `parametric` | GLM | none | deterministic |
| 2 | `gam` | smoothing-spline GAM | fixed df = 4 | deterministic |
| 3 | `lasso` | ℓ₁ GLM, 2-way interactions | λ by V-fold CV | stream-seeded |
| 4 | `knn` | k-nearest neighbors | k ∈ {5..50} by systematic CV | deterministic |
| 5 | `mlp` | 1-hidden-layer neural net | none (fixed size 4) | content-seeded |
| 6 | `bart` | Bayesian additive regression trees | none (fixed priors) | content-seeded |
| 7 | `ranger` | random forest | node size {30, 60} by CV | stream-seeded |
| 8 | `xgboost` | gradient boosting (defaults) | rounds by early stopping | stream-seeded |
| 9 | `xgboost_naimi` | gradient boosting (Naimi spec) | `min_child_weight` {30, 60} by CV | stream-seeded |
| 10 | `hal_discrete` | highly adaptive lasso | smoothness {0, 1} by CV risk | stream-seeded |
| 11 | `sl_default` | SuperLearner (tmle defaults) | NNLS weights by V-fold CV | stream-seeded |
| 12 | `sl_naimi_v1` | SuperLearner (RF+XGB+GAM) | NNLS weights | stream-seeded |
| 13 | `sl_naimi_v2` | as v1 + pairwise interactions | NNLS weights | stream-seeded |
| 14 | `sl_balzer` | SuperLearner (GLM+step+MARS+mean) | NNLS weights | stream-seeded |
| 15 | `tabpfn_v3` | tabular foundation model (cloud) | none | deterministic (`random_state=0`) |
| — | `oracle`, `oracle_notrunc` | true nuisances | — | deterministic |

### 7.1 `parametric` — logistic + linear regression

Propensity: maximum-likelihood logistic regression,

logit π(X) = α₀ + α₁·X₁ + α₂·X₂ + α₃·X₃ + α₄·X₄  (main effects).

Outcome: a single pooled Gaussian GLM,

E[Y | A, X] = β₀ + βₐ·A + β₁·X₁ + β₂·X₂ + β₃·X₃ + β₄·X₄,

scored at A = 0 and A = 1 (Section 5.2). Correctly specified in the
`simple` scenario; the canonical misspecified benchmark in `complex`. No
randomness.

### 7.2 `gam` — generalized additive model

`SuperLearner::SL.gam` called directly (no ensemble), i.e. `gam::gam` with
each continuous covariate entering as a smoothing spline s(Xⱼ, df = 4), fit
by backfitting:

logit π(X) = α₀ + s₁(X₁) + s₂(X₂) + s₃(X₃) + s₄(X₄)  (binomial)

μₐ(X) = β₀ + s₁(X₁) + s₂(X₂) + s₃(X₃) + s₄(X₄)  (Gaussian, arm-stratified).

`deg.gam = 4` follows the Zivich & Breskin (2021) DCDR configuration.
Deterministic.

### 7.3 `lasso` — ℓ₁-penalized regression with two-way interactions

Design matrix: all main effects plus all pairwise products
(`model.matrix(~ .^2)`; with p = 4 covariates, 10 columns; no squared
terms). Fits `glmnet::cv.glmnet` with α = 1 (pure lasso), binomial family
for A and Gaussian for each arm's Y, V-fold internal CV, and predictions at
λₘᵢₙ (the CV-risk-minimizing penalty):

β̂(λ) = argmin over β of { (1/n)·Σᵢ ℓ(Yᵢ, β₀ + βᵀxᵢ) + λ·‖β‖₁ },

with ℓ the negative log-likelihood of the family. Under cross-fitting the
design matrix is built once on the full data (fixing column alignment) but
the CV and fit use training rows only.

### 7.4 `knn` — k-nearest neighbors

`caret::knnreg` on standardized covariates (centering/scaling learned on the
training rows only). For a query point x, the prediction is the mean of the
response over the k nearest training points in Euclidean distance; run on
the 0/1 treatment this is the treated fraction among neighbors, an estimate
of π(x). k is selected **per nuisance** from {5, 10, 15, 20, 30, 50} by
V-fold cross-validation minimizing mean squared error (for the binary
target, the Brier score), on *systematic* folds (unit i goes to fold
i mod V), making the whole learner deterministic. Known structural property:
π̂ takes at most about k + 1 distinct values, so kNN produces coarse
propensities and extreme inverse-probability weights by construction — it is
included as a deliberately weak-for-IPW learner.

### 7.5 `mlp` — single-hidden-layer neural network

`nnet::nnet` with one hidden layer of 4 units, no weight decay, BFGS
optimization, `maxit = 250` (a documented study decision; SL.nnet's default
is 500). Covariates are standardized on training rows. The propensity net
uses a sigmoid output with **cross-entropy** loss (`entropy = TRUE`, the
Bernoulli MLE loss; SL.nnet's default squared-error-on-sigmoid occasionally
saturates to a constant). The outcome nets use a linear output on the
**z-scored** training outcome (predictions back-transformed) — without this,
the outcome scale (mean Y ≈ 120) is unreachable from nnet's ±0.7 initial
weights within the iteration cap. Each fit is seeded from a checksum of its
training data (content-seeded; Section 10), making results reproducible
regardless of run order.

### 7.6 `bart` — Bayesian additive regression trees

`dbarts::bart` (Hill 2011; Chipman, George & McCulloch sampler): a sum of 200
regression trees with regularizing priors (defaults `k = 2`, tree-depth prior
`power = 2, base = 0.95`, `numcut = 100`), 500 posterior draws after 100
burn-in. Propensity: probit BART on the binary treatment, with

π̂(x) = (1/D) · Σ Φ( f⁽ᵈ⁾(x) )  summed over posterior draws d = 1, …, D,  D = 500,

the posterior-mean probability (Φ = standard-normal CDF). Outcomes: Gaussian
BART per arm, posterior-mean predictions. No covariate standardization
(trees are scale-equivariant). Content-seeded like `mlp`. Degenerate guard:
if a training arm has only one treatment class, the propensity falls back to
the empirical mean.

### 7.7 `ranger` — random forest

`ranger` with 500 trees and `mtry = 2` (both fixed); `min.node.size`
selected **per nuisance** from {30, 60} by V-fold CV minimizing validation
MSE. Propensity: probability forest on factor(A); outcomes: regression
forests, arm-stratified. The large-node-size grid follows the Naimi et al.
ensemble spec (Section 7.12) and acts as the forest's smoothing control.

### 7.8 `xgboost` — gradient boosting, package defaults

`xgboost` at library defaults — `max_depth = 6`, `eta = 0.3`,
`subsample = 1`, `colsample_bytree = 1`, `min_child_weight = 1` — with the
number of boosting rounds chosen by `xgb.cv` early stopping (cap 500,
patience 20, V-fold; log-loss for the propensity, RMSE for the outcomes),
then refit on the full training data for the selected rounds. Objectives:
`binary:logistic` / `reg:squarederror`. Included as the
"defaults-as-an-analyst-would-run-them" contrast to the tuned Naimi spec
below.

### 7.9 `xgboost_naimi` — gradient boosting, Naimi et al. (2023) spec

Fixed 500 rounds, `max_depth = 4`, `eta = 0.1`; `min_child_weight` selected
per nuisance from {30, 60} by V-fold `xgb.cv` (minimum of the mean
validation metric over the 500-round path). The final model always runs the
full 500 rounds (no early stopping), per the published specification.

### 7.10 `hal_discrete` — highly adaptive lasso (discrete selector)

Two `hal9001::fit_hal` candidates per nuisance, both with tensor-product
basis up to interaction degree 2 over the 4 covariates and no basis
reduction: smoothness order 0 (zero-order/indicator basis — piecewise
constant) versus smoothness order 1 (first-order splines — piecewise
linear). Each candidate is an ℓ₁-penalized regression on its (large) basis
expansion with λ chosen by `hal9001`'s internal cross-validation
(package-hard-coded 10-fold at every n — the one internal-CV setting not
governed by V). The learner then acts as a **discrete SuperLearner**:
whichever candidate has the smaller internal CV risk (`min(cvm)`) supplies
the predictions. Binomial for the propensity, Gaussian arm-stratified for
outcomes. Cross-fitting runs the K folds in parallel (`mclapply`) under
deterministic L'Ecuyer-CMRG per-fold RNG streams
([`DEVIATIONS.md`](../DEVIATIONS.md) D8). Note: at n = 5,000 the *archived
production* study substituted a cheaper single-fit HAL; this codebase
implements `hal_discrete` at all n ([`DEVIATIONS.md`](../DEVIATIONS.md) D6).

### 7.11 SuperLearner ensembles — shared machinery

`sl_default`, `sl_naimi_v1`, `sl_naimi_v2`, `sl_balzer` share one recipe
([`R/learners/helpers_superlearner.R`](../R/learners/helpers_superlearner.R)):
for each nuisance, a **convex non-negative-least-squares SuperLearner**
(`method.NNLS`, the package default) with V-fold internal CV. Writing
f̂₁, …, f̂ₘ for the M candidate algorithms, the ensemble weights solve

α̂ = argmin over α (all αₘ ≥ 0) of Σᵢ ( Tᵢ − Σₘ αₘ·f̂ₘ(Xᵢ) )²,

computed on the V-fold out-of-fold (cross-validated) candidate predictions,
then normalized so Σₘ α̂ₘ = 1. Here T is the target (A under the binomial
family; arm-restricted Y under the Gaussian family), and the returned
prediction is Σₘ α̂ₘ·f̂ₘ(x) with each candidate refit on the full training
set. Fit order is fixed (propensity → μ₀ → μ₁; per fold under CF) to pin
RNG consumption.

### 7.12 `sl_naimi_v1` and `sl_naimi_v2` — the Naimi et al. ensembles

Candidate library (10 algorithms, identical for propensity and outcome):

- `SL.ranger`: 500 trees, `mtry = 2` × `min.node.size` ∈ {30, 60} (2 candidates)
- `SL.xgboost`: 500 trees, `max_depth = 4`, shrinkage 0.1 × `minobspernode` ∈ {30, 60} (2 candidates)
- `SL.gam`: smoothing splines with `deg.gam` ∈ {3, 4, 5, 6, 7, 8} (6 candidates)

`sl_naimi_v1` fits this library on the covariates as supplied (main terms).
`sl_naimi_v2` fits the identical library on the covariates **augmented with
all pairwise interaction columns** Xᵢ·Xⱼ, i < j (6 extra columns at p = 4),
for both nuisances. All algorithm-level hyperparameters match the published
Naimi et al. specification
([`DEVIATIONS_vs_upstream.md`](../DEVIATIONS_vs_upstream.md) §A7).
**Manuscript naming note:** in the paper's own numbering these correspond to
"Version 2" (RF + XGB + GAM) and "Version 2 + interactions" respectively —
the paper's GAM-free "Version 1" was deliberately not implemented — so these
ensembles should be described by composition, never as the paper's
"Version 1" ([`DEVIATIONS_vs_upstream.md`](../DEVIATIONS_vs_upstream.md) §C3).

### 7.13 `sl_default` — the tmle package's default libraries

The default SuperLearner libraries of `tmle::tmle()` 2.x, fit through the
shared recipe: propensity library `SL.glm + tmle.SL.dbarts.k.5 + SL.gam`;
outcome library `SL.glm + tmle.SL.dbarts2 + SL.glmnet`. This arm answers
"what would an analyst get from the tmle package defaults?", while keeping
the estimator layer identical to every other learner.

### 7.14 `sl_balzer` — parsimonious ensemble

Library `SL.glm` (main-effects GLM) + `SL.step.interaction` (stepwise
selection over main effects and two-way interactions by AIC) +
`SL.earth` (MARS — multivariate adaptive regression splines) + `SL.mean`
(the constant model, as a floor), per Balzer & Westling (2023).

### 7.15 `tabpfn_v3` — tabular foundation model (cloud track)

TabPFN version 3 (Hollmann et al.) via the cloud API
(`tabpfn-client == 0.3.0`, pinned), the study's prior-fitted
transformer-foundation-model arm: a pretrained transformer performing
in-context (training-free) prediction, with `n_estimators = 8` ensemble
members and `random_state = 0` (deterministic within a run window; the cloud
endpoint drifts across weeks — see §10). Propensity:
`TabPFNClassifier` class-1 probability; outcomes: `TabPFNRegressor`,
arm-stratified — exactly mirroring the R learners' contract. The Python step
([`python/tabpfn_v3_nuisance.py`](../python/tabpfn_v3_nuisance.py)) reads the
R-exported datasets and the R-exported shared fold assignment (so the CF arm
uses the *same* folds as every R learner) and writes raw nuisance vectors;
stage 2b applies the study truncation and the identical four R estimators —
the TabPFN arm differs from the R learners in nothing but the nuisance model.
Cloud calls are wrapped in a transient-failure retry with deterministic
backoff, which cannot change any estimate (fixed `random_state`;
[`DEVIATIONS.md`](../DEVIATIONS.md) D10).

## 8. Oracle reference arms

Handled directly by stage 2 ([`R/stage2_estimate.R`](../R/stage2_estimate.R)),
because they require the DGP truth that fitted learners never see. Both arms
substitute the true unit-level nuisances π∗(C), μ₀∗(C), μ₁∗(C)
(Section 3.1) for the fitted ones and run the same four estimators — the
attainable performance ceiling, isolating estimator-level (finite-sample,
weighting, targeting) behavior from nuisance-estimation error. Note the
oracle nuisances are the truth as functions of the *confounders*; the
scenario label only changes the covariate frame passed to the estimators
(used by `tmle()` as `W`), not the nuisance values, so oracle rows are
effectively scenario-invariant.

- **`oracle`**: true propensities truncated to [0.025, 0.975] — the
  apples-to-apples ceiling under the study truncation policy.
- **`oracle_notrunc`**: raw true propensities for IPW and AIPW
  (g-computation uses only μ∗ and is unaffected). Its TMLE is re-bounded
  to [0.025, 0.975] by the explicit `gbound` and is therefore identical to
  `oracle`'s TMLE ([`DEVIATIONS.md`](../DEVIATIONS.md) D2).

Cross-fitting is a no-op for known truth, so identical oracle rows are
written under both cross-fitting labels (one computation per simulation).
Any "effect of cross-fitting averaged over learners" summary must therefore
exclude the oracle arms.

## 9. Performance metrics

Aggregation: [`R/stage3_summarize.R`](../R/stage3_summarize.R). Within each
cell (sample size × scenario × learner × cross-fitting × estimator), over the
S non-failed simulations (failures are counted and reported, never silently
dropped), with ψ̂ₛ the estimate in simulation s, SEₛ its reported analytic
standard error, and ψ₀ = 6:

| Metric | Definition |
|---|---|
| Mean bias | b̄ = (1/S) · Σₛ ( ψ̂ₛ − ψ₀ ) |
| Monte-Carlo SE of bias | empirical SE / √S |
| Empirical SE | SD of ψ̂ₛ across the S simulations |
| Mean model SE | (1/S) · Σₛ SEₛ (the average reported analytic SE) |
| SE ratio | mean model SE / empirical SE (calibration of the analytic SE; < 1 = anti-conservative) |
| RMSE | √[ (1/S) · Σₛ ( ψ̂ₛ − ψ₀ )² ] |
| Coverage | (1/S) · Σₛ 1{ CI of sim s contains ψ₀ }, nominal 0.95 |
| MC SE of coverage | √[ p̂·(1 − p̂)/S ] (binomial; ≈ 1.5 pp at 95% with S = 200) |

G-computation has no CI, so its coverage (and coverage MCSE) is `NA` rather
than an average over a partial set. Nuisance-fit CPU time is averaged per
cell; timing comparability caveats are documented in
[`ISSUES_AND_IMPROVEMENTS.md`](ISSUES_AND_IMPROVEMENTS.md) §13.

## 10. Randomness, seeds, and reproducibility

All RNG rules are centralized in
[`R/folds_and_seeds.R`](../R/folds_and_seeds.R). With
`sim_seed = starting_seed + sim_id` (`starting_seed = 1`; sims 1..200 →
seeds 2..201 at every sample size):

| Draw | Seed | Scope |
|---|---|---|
| Dataset (confounders, treatment, errors) | `set.seed(sim_seed)` | Section 3; fixed draw order |
| CF fold partition | `sim_seed × 1000 + 7` | learner-independent → shared folds; also pins the RNG stream for the CF fits that follow |
| Full-sample (no-CF) fits | `sim_seed × 1000 + 11` | distinct stream from CF |
| `mlp`, `bart` model fits | checksum of the training (Y, A, X) | content-seeded per fit; ambient RNG state restored afterwards |
| `hal_discrete` CF fold workers | L'Ecuyer-CMRG streams from the ambient (fold-seed-pinned) state | deterministic across runs and across core counts |
| TabPFN v3 | `random_state = 0`, `n_estimators = 8` | deterministic **within a run window only** — see below |

Deterministic learners (`parametric`, `gam`, `knn`) consume no randomness.
Because every (learner, simulation, path) re-seeds, any subset of simulations
can be re-run in any order and reproduces identical numbers **for every R
learner**; stage 2 is resume-safe (completed simulations are skipped, failed
ones re-queued).

**TabPFN v3 qualifier (correction 2026-07-26).** `random_state = 0` makes
cloud fits reproducible against contemporaneous calls, but the served model
drifts over time: re-fitting archived May-2026 production inputs on
2026-07-24 moved per-sim ATEs by up to 0.16 — including 0.034 on
G-computation, which involves no propensity, truncation, or seed — so the
drift is server-side (evidence:
`~/Documents/Thesis_Aim2/tmleR_extension_n1200_n5000_PLAN.md`, REVISION v2).
Archived TabPFN results are therefore reproducible only from their **saved
nuisance fits** (the [`VALIDATION.md`](VALIDATION.md) §3 route), and any
manuscript reproducibility statement about the TabPFN arm must carry this
caveat.
Bit-level reproduction of RNG-dependent learners additionally assumes the
same package versions (a package changing its internal RNG consumption breaks
bit-agreement without invalidating the statistics). Versions used for
validation — R 4.5.2; tmle 2.1.1, SuperLearner 2.0.40, sandwich 3.1.1,
ranger 0.18.0, xgboost 3.1.3.1, glmnet 4.1.10, gam 1.22.7, earth 5.3.5,
nnet 7.3.20, caret 7.0.1, dbarts 0.9.32, hal9001 0.4.6 — are recorded in
[`VALIDATION.md`](VALIDATION.md), which also documents byte-identical
reproduction of the production datasets/folds at all three sample sizes and
exact (≤ ~5×10⁻¹⁴) reproduction of archived per-simulation estimator rows for
the validated cells, with the known exceptions catalogued in
[`DEVIATIONS.md`](../DEVIATIONS.md) (unseeded historical n=1200 no-CF runs;
historically irreproducible `hal_discrete` CF; TabPFN pairing and TMLE
fluctuation differences at n=1200/5000).

## 11. Relationship to Naimi, Mishler & Kennedy (2023)

Faithfully reproduced and verified against the public materials: the four
estimator formulas; the pooled-parametric / arm-stratified-ML outcome
convention; the DGP functional form and the paper's propensity model; the
inner CV fold schedule (10/5/5); the [0.025, 0.975] truncation value; DML2
pooled cross-fitting with single-step TMLE targeting; and all SuperLearner
ensemble hyperparameters. Declared deviations: the explicit TMLE `gbound`
(removes a truncation inconsistency latent in the upstream code at n = 200)
and distributional — not per-simulation — reproduction (different seeds and
draw mechanics). Study choices not verifiable upstream: the outer cross-fit
fold counts and every learner beyond the parametric GLM and the two
RF+XGB+GAM ensembles (those additional learners are extensions, not
reproductions). Full itemization with evidence:
[`DEVIATIONS_vs_upstream.md`](../DEVIATIONS_vs_upstream.md).

## References

- Naimi AI, Mishler AE, Kennedy EH (2023). Challenges in obtaining valid
  causal effect estimates with machine learning algorithms. *Am J Epidemiol*
  192(9):1536–44.
- Kang JDY, Schafer JL (2007). Demystifying double robustness. *Stat Sci*
  22(4):523–39.
- Chernozhukov V, Chetverikov D, Demirer M, Duflo E, Hansen C, Newey W,
  Robins J (2018). Double/debiased machine learning. *Econom J* 21(1):C1–C68.
- Zivich PN, Breskin A (2021). Machine learning for causal inference: on the
  use of cross-fit estimators. *Epidemiology* 32(3):393–401.
- Balzer LB, Westling T (2023). Demystifying statistical inference when using
  machine learning in causal research. *Am J Epidemiol* 192(9):1545–9.
- Hill JL (2011). Bayesian nonparametric modeling for causal inference.
  *J Comput Graph Stat* 20(1):217–40.
- Gruber S, van der Laan MJ (2010). A targeted maximum likelihood estimator
  of a causal effect on a bounded continuous outcome. *Int J Biostat* 6(1).
- van der Laan MJ, Polley EC, Hubbard AE (2007). Super Learner.
  *Stat Appl Genet Mol Biol* 6(1).
- Benkeser D, van der Laan MJ (2016). The highly adaptive lasso estimator.
  *Proc IEEE Int Conf Data Sci Adv Anal* 689–96.
- Hollmann N, Müller S, et al. (2025). TabPFN (model version 3, cloud API).
