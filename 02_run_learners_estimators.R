# =============================================================================
# 02_run_learners_estimators.R — nuisance learners + the four ATE estimators
#
# For every (scenario x learner x cross-fit x sim) cell: estimate the three
# nuisance functions (propensity score + arm-specific outcome regressions),
# then run IPW / G-computation / AIPW / TMLE on those same nuisance vectors.
# Per-sim results append to results/n<k>/per_config/ (resume-safe).
#
#   Rscript 02_run_learners_estimators.R --n 1200                  # R learners + oracle
#   Rscript 02_run_learners_estimators.R --n 1200 --learners parametric,ranger --sims 1:20
#   python  python/tabpfn_v3_nuisance.py --n 1200                  # TabPFN fits, then:
#   Rscript 02_run_learners_estimators.R --n 1200 --track tabpfn   # estimators over them
#
# Learner roster (the manuscript's names in parentheses):
#   parametric (Parametric)   gam (GAM)            lasso (LASSO)
#   knn (KNN)                 bart (BART)          ranger (Random Forest)
#   xgboost_naimi (XGBoost)   hal_discrete (HAL, n=200/1200)
#   hal_n5000 (HAL, n=5000)   sl_default (SL Default)
#   sl_naimi_v2 (SL Naimi)    sl_balzer (SL Balzer)
#   tabpfn_v3 / tabpfn_v3_ne1 (TabPFN / TabPFN no aggregation; via --track tabpfn)
#   oracle (Oracle reference arm: true nuisances, truncated)
# =============================================================================
# --- Shared command-line / configuration header ------------------------------
# Usage (all scripts):  Rscript <script> --n <200|1200|5000> [options]
#   --n <k>          sample size (required)
#   --sims a:b|a,b,c|k   simulation subset (default: all 200)
#   --scenarios s1,s2    subset of simple,complex
#   --learners l1,l2     subset of the learner roster (script 02 only)
#   --cross_fit true,false  subset of cross-fitting arms (script 02 only)
#   --track r|tabpfn|all    which estimation track to run (script 02 only;
#                           default r = the R learners + oracle. tabpfn = the
#                           estimators over saved TabPFN nuisance vectors)

suppressPackageStartupMessages(library(yaml))

PROJECT_ROOT <- {
  cmd_args <- commandArgs(trailingOnly = FALSE)
  f <- grep("--file=", cmd_args, value = TRUE)
  if (length(f) > 0) dirname(normalizePath(sub("--file=", "", f))) else getwd()
}

CONFIG <- yaml::yaml.load_file(file.path(PROJECT_ROOT, "config.yaml"))

args <- commandArgs(trailingOnly = TRUE)
OPTS <- list(n = NULL, learners = NULL, sims = NULL,
             scenarios = NULL, cross_fit = NULL, track = "r")
i <- 1
while (i <= length(args)) {
  key <- args[i]; val <- if (i < length(args)) args[i + 1] else NA
  switch(key,
    "--n"         = { OPTS$n         <- as.integer(val) },
    "--learners"  = { OPTS$learners  <- strsplit(val, ",")[[1]] },
    "--scenarios" = { OPTS$scenarios <- strsplit(val, ",")[[1]] },
    "--cross_fit" = { OPTS$cross_fit <- as.logical(strsplit(val, ",")[[1]]) },
    "--track"     = { OPTS$track     <- val },
    "--sims"      = {
      OPTS$sims <- if (grepl(":", val)) {
        p <- as.integer(strsplit(val, ":")[[1]]); p[1]:p[2]
      } else if (grepl(",", val)) {
        unique(as.integer(strsplit(val, ",")[[1]]))
      } else {
        1:as.integer(val)
      }
    },
    stop(sprintf("Unknown argument: %s", key))
  )
  i <- i + 2
}

if (is.null(OPTS$n)) stop("--n is required (one of: ",
                          paste(CONFIG$simulation$sample_sizes, collapse = ", "), ")")
if (!OPTS$n %in% CONFIG$simulation$sample_sizes)
  stop(sprintf("--n %d not in config sample_sizes", OPTS$n))
if (!OPTS$track %in% c("r", "tabpfn", "all"))
  stop("--track must be one of: r, tabpfn, all")

# The manuscript's "HAL" is a sample-size-dependent variant (see README):
# hal_discrete at n=200/1200, the simplified single-fit hal_n5000 at n=5000.
default_roster <- c(unlist(CONFIG$learners),
                    CONFIG$estimation$hal_by_n[[as.character(OPTS$n)]])
if (is.null(OPTS$learners))  OPTS$learners  <- default_roster
if (is.null(OPTS$sims))      OPTS$sims      <- 1:CONFIG$simulation$n_sims
if (is.null(OPTS$scenarios)) OPTS$scenarios <- CONFIG$simulation$scenarios
if (is.null(OPTS$cross_fit)) OPTS$cross_fit <- unlist(CONFIG$estimation$cross_fit_options)

CONFIG$paths <- lapply(CONFIG$paths, function(p) file.path(PROJECT_ROOT, p))
CONFIG$resolved <- list(
  n         = OPTS$n,
  n_splits  = CONFIG$estimation$folds_by_n[[as.character(OPTS$n)]],
  cv_folds  = CONFIG$estimation$folds_by_n[[as.character(OPTS$n)]],
  data_dir  = file.path(CONFIG$paths$data_inputs,    sprintf("n%d", OPTS$n)),
  export_dir = file.path(CONFIG$paths$data_processed, sprintf("n%d", OPTS$n)),
  results_dir = file.path(CONFIG$paths$results,      sprintf("n%d", OPTS$n))
)
if (is.null(CONFIG$resolved$n_splits))
  stop(sprintf("No folds_by_n entry for n=%d in config.yaml", OPTS$n))
# -----------------------------------------------------------------------------


# =============================================================================
# ==== RNG rules (fold + fit seeds; sim_seed = starting_seed + sim_id)
# =============================================================================

# =============================================================================
# R/folds_and_seeds.R — every RNG rule in the study, in one place
#
# Seed derivations (sim_seed = starting_seed + sim_id, from config.yaml):
#
#   data          set.seed(sim_seed)              in simulate_data()
#   CF folds      set.seed(sim_seed * 1000 + 7)   in generate_folds(); the same
#                                                 folds are shared by EVERY
#                                                 learner (head-to-head pairing),
#                                                 and this set.seed also pins the
#                                                 RNG stream for the CF learner
#                                                 fits that follow
#   no-CF fits    set.seed(sim_seed * 1000 + 11)  before each full-sample fit
#                                                 (distinct stream from CF)
#
# Two learners additionally use content-derived seeds so each individual model
# fit is reproducible regardless of ambient RNG state: mlp and bart hash
# (Y, A, X) to a seed, fit under it, and restore the previous RNG state
# (see their files).
#
# The Python TabPFN track needs no RNG of its own: it reads the R-exported
# datasets AND the R-exported fold assignments (stage 1), and TabPFN v3 is
# deterministic at random_state = 0.
# =============================================================================

#' Balanced random fold assignment (1..n_splits), deterministic given seed
generate_folds <- function(n, n_splits, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  sample(rep(1:n_splits, ceiling(n / n_splits))[1:n])
}

cf_fold_seed <- function(sim_seed) sim_seed * 1000 + 7
nocf_seed    <- function(sim_seed) sim_seed * 1000 + 11


# =============================================================================
# ==== Data-generating process (needed to read the cached ground truth)
# =============================================================================

# =============================================================================
# R/dgp.R — data-generating process
#
# DGP described by Naimi, Mishler & Kennedy (2023), adapted from Kang & Schafer (2007:
# 4 iid standard-normal confounders C1-C4, binary treatment (logistic in C),
# continuous outcome (linear in C, homogeneous/fixed effect), true ATE = 6.
#
# The "complex" scenario hands the analyst the Kang-Schafer transforms
# Z = h(C) instead of C, so no learner ever sees the correctly specified
# covariates:
#   Z1 = exp(C1/2)
#   Z2 = C2 / (1 + exp(C1)) + 10
#   Z3 = (C1*C3/25 + 0.6)^3
#   Z4 = (C2 + C4 + 20)^2
#
# Reproducibility: simulate_data(n, seed, ...) is fully deterministic given
# `seed`. The study convention is seed = starting_seed + sim_id, so sims
# 1..200 use seeds 2..201 at every sample size. Draw order (confounders ->
# treatment -> outcome errors) is fixed and must not be reordered: it defines
# the datasets every archived result was computed on.
# =============================================================================

generate_confounders <- function(n) {
  C <- matrix(rnorm(n * 4), nrow = n, ncol = 4)
  colnames(C) <- paste0("C", 1:4)
  C
}

transform_confounders <- function(C) {
  Z <- matrix(0, nrow = nrow(C), ncol = 4)
  colnames(Z) <- paste0("Z", 1:4)
  Z[, 1] <- exp(C[, 1] / 2)
  Z[, 2] <- C[, 2] / (1 + exp(C[, 1])) + 10
  Z[, 3] <- (C[, 1] * C[, 3] / 25 + 0.6)^3
  Z[, 4] <- (C[, 2] + C[, 4] + 20)^2
  Z
}

generate_propensity <- function(C, beta_a) {
  as.vector(plogis(cbind(1, C) %*% beta_a))
}

generate_treatment <- function(pi) {
  rbinom(length(pi), 1, pi)
}

generate_outcomes <- function(C, A, beta_y, effect_size, error_sd) {
  epsilon <- rnorm(nrow(C), 0, error_sd)
  mu0 <- cbind(1, C) %*% beta_y
  Y0  <- mu0 + epsilon
  Y1  <- mu0 + effect_size + epsilon
  list(Y   = as.vector(A * Y1 + (1 - A) * Y0),
       Y0  = as.vector(Y0),
       Y1  = as.vector(Y1),
       mu0 = as.vector(mu0),
       mu1 = as.vector(mu0 + effect_size))
}

#' Generate one complete simulated dataset (deterministic given seed)
simulate_data <- function(n, seed, dgp_params) {
  set.seed(seed)
  beta_a      <- unlist(dgp_params$beta_a)
  beta_y      <- unlist(dgp_params$beta_y)
  effect_size <- dgp_params$true_ate
  error_sd    <- dgp_params$error_sd

  C        <- generate_confounders(n)
  Z        <- transform_confounders(C)
  pi_true  <- generate_propensity(C, beta_a)
  A        <- generate_treatment(pi_true)
  outcomes <- generate_outcomes(C, A, beta_y, effect_size, error_sd)

  list(C = C, Z = Z, A = A, Y = outcomes$Y,
       data_simple  = data.frame(C, A = A, Y = outcomes$Y),
       data_complex = data.frame(Z, A = A, Y = outcomes$Y),
       pi_true  = pi_true,
       mu0_true = outcomes$mu0,
       mu1_true = outcomes$mu1,
       Y0 = outcomes$Y0, Y1 = outcomes$Y1,
       n = n, seed = seed, true_ate = effect_size)
}


# =============================================================================
# ==== Nuisance dispatcher + learner registry
# =============================================================================

# --- Nuisance estimation dispatcher ------------------------------------------
# Each learner section below defines
#   fit_nuisance_<name>(Y, A, X, config)                -> list(pihat, mu0hat, mu1hat)
#   fit_nuisance_cf_<name>(Y, A, X, fold_ids, config)   -> same, out-of-fold
# and registers itself in LEARNERS. Both functions must truncate pihat to
# config$pi_bounds before returning.
#
# Cross-fitting convention: DML2 / pooled. The CF path returns full-length
# out-of-fold nuisance vectors; each estimator is then solved ONCE on the
# pooled vectors (no per-fold ATE averaging). Fold assignments come from
# generate_folds() with a learner-INDEPENDENT seed, so all learners share the
# same folds within a simulation and remain per-sim paired.
#
# There is deliberately no generic CF fallback: a learner without an explicit
# fit_nuisance_cf_<name> cannot be cross-fit (a historical fallback that
# recycled training-fold predictions produced silently wrong CF results and
# was removed from the original pipeline in 2026-05).

LEARNERS <- list()

register_learner <- function(name, fit_nuisance, fit_nuisance_cf = NULL,
                             packages = character(0)) {
  LEARNERS[[name]] <<- list(name = name,
                            fit_nuisance = fit_nuisance,
                            fit_nuisance_cf = fit_nuisance_cf,
                            packages = packages)
  invisible(NULL)
}

#' Estimate nuisance functions for one (learner, dataset, cross_fit) triple
#'
#' Seeding (see the RNG rules section above): the CF path is pinned by the
#' fold seed (sim_seed*1000+7), the no-CF path by sim_seed*1000+11. Every run
#' of the same triple is therefore bit-reproducible.
estimate_nuisance <- function(learner_name, Y, A, X, cross_fit, config, sim_seed) {
  learner <- LEARNERS[[learner_name]]
  if (is.null(learner)) stop(sprintf("Learner '%s' not loaded", learner_name))
  # hal_n5000's cross-fit path derives its per-fold seeds from the sim seed.
  config$sim_seed <- sim_seed

  pt <- proc.time()
  if (cross_fit) {
    if (is.null(learner$fit_nuisance_cf))
      stop(sprintf("Learner '%s' has no fit_nuisance_cf implementation", learner_name))
    fold_ids <- generate_folds(length(Y), config$n_splits,
                               seed = cf_fold_seed(sim_seed))
    result <- learner$fit_nuisance_cf(Y, A, X, fold_ids, config)
  } else {
    set.seed(nocf_seed(sim_seed))
    result <- learner$fit_nuisance(Y, A, X, config)
  }
  el <- proc.time() - pt

  # CPU = parent + forked children (captures mclapply CF learners).
  result$nuisance_time      <- el[["user.self"]] + el[["sys.self"]] +
                               el[["user.child"]] + el[["sys.child"]]
  result$nuisance_time_wall <- el[["elapsed"]]
  result$learner   <- learner_name
  result$cross_fit <- cross_fit
  # Carried to estimate_tmle() so tmle() applies the study truncation policy
  # instead of its internal adaptive default.
  result$pi_bounds <- config$pi_bounds
  result
}


# =============================================================================
# ==== Shared SuperLearner helpers
# =============================================================================

# =============================================================================
# helpers_superlearner.R — shared machinery for the SuperLearner-based learners
# (sl_default, sl_naimi_v1, sl_naimi_v2, sl_balzer)
#
# Every SL learner follows the same recipe: convex NNLS SuperLearner
# (method.NNLS, V = config$cv_folds) for the propensity (binomial) and for
# each treatment arm's outcome model (gaussian, arm-stratified). Fit order is
# always propensity -> mu0 -> mu1 (per fold under CF), which fixes the RNG
# consumption order.
#
# ps_library / out_library may differ (sl_default); transform_X lets a learner
# augment the covariates first (sl_naimi_v2's pairwise interactions).
# =============================================================================

suppressPackageStartupMessages(library(SuperLearner))

sl_fit_nuisance <- function(Y, A, X, config, ps_library,
                            out_library = ps_library,
                            transform_X = identity) {
  X <- transform_X(as.data.frame(X))
  bounds <- config$pi_bounds
  cv_folds <- config$cv_folds

  ps_fit <- SuperLearner(Y = A, X = X, family = binomial(),
                         SL.library = ps_library,
                         cvControl = list(V = cv_folds))
  pihat <- pmax(bounds[1], pmin(bounds[2], as.numeric(ps_fit$SL.predict)))

  idx_0 <- A == 0; idx_1 <- A == 1
  Q0_fit <- SuperLearner(Y = Y[idx_0], X = X[idx_0, , drop = FALSE],
                         newX = X, family = gaussian(),
                         SL.library = out_library,
                         cvControl = list(V = cv_folds))
  Q1_fit <- SuperLearner(Y = Y[idx_1], X = X[idx_1, , drop = FALSE],
                         newX = X, family = gaussian(),
                         SL.library = out_library,
                         cvControl = list(V = cv_folds))

  list(pihat = pihat,
       mu0hat = as.numeric(Q0_fit$SL.predict),
       mu1hat = as.numeric(Q1_fit$SL.predict))
}

sl_fit_nuisance_cf <- function(Y, A, X, fold_ids, config, ps_library,
                               out_library = ps_library,
                               transform_X = identity) {
  X <- transform_X(as.data.frame(X))
  n <- length(Y)
  bounds <- config$pi_bounds
  cv_folds <- config$cv_folds
  K <- max(fold_ids)

  pihat <- mu0hat <- mu1hat <- rep(NA_real_, n)
  for (k in 1:K) {
    train_idx <- which(fold_ids != k)
    test_idx  <- which(fold_ids == k)
    X_train <- X[train_idx, , drop = FALSE]
    X_test  <- X[test_idx,  , drop = FALSE]
    Y_train <- Y[train_idx]
    A_train <- A[train_idx]

    ps_fit <- SuperLearner(Y = A_train, X = X_train, newX = X_test,
                           family = binomial(), SL.library = ps_library,
                           cvControl = list(V = cv_folds))
    pihat[test_idx] <- as.numeric(ps_fit$SL.predict)

    i0 <- which(A_train == 0)
    Q0_fit <- SuperLearner(Y = Y_train[i0], X = X_train[i0, , drop = FALSE],
                           newX = X_test, family = gaussian(),
                           SL.library = out_library,
                           cvControl = list(V = cv_folds))
    mu0hat[test_idx] <- as.numeric(Q0_fit$SL.predict)

    i1 <- which(A_train == 1)
    Q1_fit <- SuperLearner(Y = Y_train[i1], X = X_train[i1, , drop = FALSE],
                           newX = X_test, family = gaussian(),
                           SL.library = out_library,
                           cvControl = list(V = cv_folds))
    mu1hat[test_idx] <- as.numeric(Q1_fit$SL.predict)
  }

  pihat <- pmax(bounds[1], pmin(bounds[2], pihat))
  list(pihat = pihat, mu0hat = mu0hat, mu1hat = mu1hat)
}

# --- Naimi v1/v2 shared candidate library ------------------------------------
# 10 candidates via create.Learner (registered once, in the global env):
#   SL.ranger(500 trees, mtry 2) x min.node.size {30, 60}
#   SL.xgboost(500 trees, depth 4, shrinkage 0.1) x minobspernode {30, 60}
#   SL.gam x deg.gam 3:8
create_naimi_library <- function() {
  if (!exists("SL_NAIMI_V1_CREATED", envir = .GlobalEnv)) {
    rf_learner <- create.Learner("SL.ranger",
      params = list(num.trees = 500, mtry = 2),
      tune = list(min.node.size = c(30, 60)),
      name_prefix = "RF", env = .GlobalEnv)
    xgb_learner <- create.Learner("SL.xgboost",
      params = list(ntrees = 500, max_depth = 4, shrinkage = 0.1),
      tune = list(minobspernode = c(30, 60)),
      name_prefix = "XGB", env = .GlobalEnv)
    gam_learner <- create.Learner("SL.gam",
      tune = list(deg.gam = 3:8),
      name_prefix = "GAM", env = .GlobalEnv)
    assign("RF_LEARNER_NAMES",  rf_learner$names,  envir = .GlobalEnv)
    assign("XGB_LEARNER_NAMES", xgb_learner$names, envir = .GlobalEnv)
    assign("GAM_LEARNER_NAMES", gam_learner$names, envir = .GlobalEnv)
    assign("SL_NAIMI_V1_CREATED", TRUE, envir = .GlobalEnv)
  }
  c(get("RF_LEARNER_NAMES",  envir = .GlobalEnv),
    get("XGB_LEARNER_NAMES", envir = .GlobalEnv),
    get("GAM_LEARNER_NAMES", envir = .GlobalEnv))
}

#' Augment a covariate frame with all pairwise interaction columns
add_pairwise_interactions <- function(X) {
  X <- as.data.frame(X)
  p <- ncol(X)
  if (p < 2) return(X)
  orig_names <- colnames(X)
  interactions <- list()
  for (i in 1:(p - 1)) {
    for (j in (i + 1):p) {
      interactions[[paste0(orig_names[i], "_x_", orig_names[j])]] <- X[[i]] * X[[j]]
    }
  }
  cbind(X, as.data.frame(interactions))
}


# =============================================================================
# ==== Learner: parametric
# =============================================================================

# =============================================================================
# parametric — logistic regression (PS) + single linear regression (outcome)
#
# The outcome model is the ONE non-arm-stratified fit in the study: a single
# Y ~ A + X regression scored at A=0/A=1 (classical parametric g-computation).
# Deterministic (no RNG).
# =============================================================================

fit_nuisance_parametric <- function(Y, A, X, config) {
  X <- as.data.frame(X)
  bounds <- config$pi_bounds
  cov_names <- colnames(X)

  ps_dat <- data.frame(A = A, X)
  ps_formula <- as.formula(paste("A ~", paste(cov_names, collapse = " + ")))
  ps_model <- glm(ps_formula, family = binomial(link = "logit"), data = ps_dat)
  pihat <- pmax(bounds[1], pmin(bounds[2], predict(ps_model, type = "response")))

  out_dat <- data.frame(Y = Y, A = A, X)
  out_formula <- as.formula(paste("Y ~ A +", paste(cov_names, collapse = " + ")))
  out_model <- glm(out_formula, family = gaussian, data = out_dat)

  dat_0 <- out_dat; dat_0$A <- 0
  dat_1 <- out_dat; dat_1$A <- 1
  list(pihat = pihat,
       mu0hat = predict(out_model, newdata = dat_0),
       mu1hat = predict(out_model, newdata = dat_1))
}

fit_nuisance_cf_parametric <- function(Y, A, X, fold_ids, config) {
  n <- length(Y)
  X <- as.data.frame(X)
  bounds <- config$pi_bounds
  cov_names <- colnames(X)
  K <- max(fold_ids)

  pihat <- mu0hat <- mu1hat <- rep(NA_real_, n)
  ps_formula  <- as.formula(paste("A ~", paste(cov_names, collapse = " + ")))
  out_formula <- as.formula(paste("Y ~ A +", paste(cov_names, collapse = " + ")))

  for (k in 1:K) {
    train_idx <- which(fold_ids != k)
    test_idx  <- which(fold_ids == k)
    train_dat <- data.frame(Y = Y[train_idx], A = A[train_idx],
                            X[train_idx, , drop = FALSE])
    test_dat  <- data.frame(Y = Y[test_idx], A = A[test_idx],
                            X[test_idx, , drop = FALSE])

    ps_model <- glm(ps_formula, family = binomial(link = "logit"), data = train_dat)
    pihat[test_idx] <- predict(ps_model, newdata = test_dat, type = "response")

    out_model <- glm(out_formula, family = gaussian, data = train_dat)
    test_0 <- test_dat; test_0$A <- 0
    test_1 <- test_dat; test_1$A <- 1
    mu0hat[test_idx] <- predict(out_model, newdata = test_0)
    mu1hat[test_idx] <- predict(out_model, newdata = test_1)
  }

  pihat <- pmax(bounds[1], pmin(bounds[2], pihat))
  list(pihat = pihat, mu0hat = mu0hat, mu1hat = mu1hat)
}

register_learner("parametric", fit_nuisance_parametric, fit_nuisance_cf_parametric)


# =============================================================================
# ==== Learner: gam
# =============================================================================

# =============================================================================
# gam — generalized additive model via SuperLearner's SL.gam wrapper
# (Zivich & Breskin DCDR configuration: deg.gam = 4)
#
# SL.gam is called directly (no ensemble): gam-package backfitting, each
# continuous covariate enters as s(x, df = 4). Propensity: binomial;
# outcome: gaussian, arm-stratified. Deterministic (no RNG).
# =============================================================================

suppressPackageStartupMessages({ library(SuperLearner); library(gam) })

.gam_fit <- function(Ytr, Xtr, Xeval, family, deg) {
  out <- SL.gam(Y = Ytr, X = as.data.frame(Xtr), newX = as.data.frame(Xeval),
                family = family, obsWeights = rep(1, length(Ytr)), deg.gam = deg)
  as.numeric(out$pred)
}

fit_nuisance_gam <- function(Y, A, X, config) {
  X <- as.data.frame(X)
  bounds <- config$pi_bounds
  deg <- 4L

  pihat <- pmax(bounds[1], pmin(bounds[2],
            .gam_fit(as.numeric(A), X, X, binomial(), deg)))

  idx0 <- A == 0; idx1 <- A == 1
  mu0hat <- .gam_fit(Y[idx0], X[idx0, , drop = FALSE], X, gaussian(), deg)
  mu1hat <- .gam_fit(Y[idx1], X[idx1, , drop = FALSE], X, gaussian(), deg)

  list(pihat = pihat, mu0hat = mu0hat, mu1hat = mu1hat)
}

fit_nuisance_cf_gam <- function(Y, A, X, fold_ids, config) {
  n <- length(Y)
  X <- as.data.frame(X)
  bounds <- config$pi_bounds
  deg <- 4L
  K <- max(fold_ids)

  pihat <- mu0hat <- mu1hat <- rep(NA_real_, n)
  for (k in 1:K) {
    tr <- which(fold_ids != k); te <- which(fold_ids == k)
    Xtr <- X[tr, , drop = FALSE]; Xte <- X[te, , drop = FALSE]
    Ytr <- Y[tr]; Atr <- A[tr]

    pihat[te] <- .gam_fit(as.numeric(Atr), Xtr, Xte, binomial(), deg)

    i0 <- Atr == 0; i1 <- Atr == 1
    mu0hat[te] <- .gam_fit(Ytr[i0], Xtr[i0, , drop = FALSE], Xte, gaussian(), deg)
    mu1hat[te] <- .gam_fit(Ytr[i1], Xtr[i1, , drop = FALSE], Xte, gaussian(), deg)
  }

  pihat <- pmax(bounds[1], pmin(bounds[2], pihat))
  list(pihat = pihat, mu0hat = mu0hat, mu1hat = mu1hat)
}

register_learner("gam", fit_nuisance_gam, fit_nuisance_cf_gam,
                 packages = c("SuperLearner", "gam"))


# =============================================================================
# ==== Learner: lasso
# =============================================================================

# =============================================================================
# lasso — cv.glmnet (alpha = 1) at lambda.min over main effects + all 2-way
# interactions (no squared terms; with 4 covariates the design has 10 columns).
# Propensity: binomial; outcome: gaussian, arm-stratified.
# Internal CV folds = config$cv_folds (RNG-dependent fold assignment; seeded
# upstream by the dispatcher).
# =============================================================================

suppressPackageStartupMessages(library(glmnet))

.lasso_design <- function(X) {
  mm <- stats::model.matrix(~ .^2, data = as.data.frame(X))
  mm[, colnames(mm) != "(Intercept)", drop = FALSE]
}

fit_nuisance_lasso <- function(Y, A, X, config) {
  bounds <- config$pi_bounds
  nf <- as.integer(config$cv_folds)
  XD <- .lasso_design(X)

  ps <- glmnet::cv.glmnet(XD, as.numeric(A), family = "binomial", alpha = 1, nfolds = nf)
  pihat <- pmax(bounds[1], pmin(bounds[2],
            as.numeric(predict(ps, newx = XD, s = "lambda.min", type = "response"))))

  idx0 <- A == 0; idx1 <- A == 1
  q0 <- glmnet::cv.glmnet(XD[idx0, , drop = FALSE], Y[idx0], family = "gaussian", alpha = 1, nfolds = nf)
  q1 <- glmnet::cv.glmnet(XD[idx1, , drop = FALSE], Y[idx1], family = "gaussian", alpha = 1, nfolds = nf)
  mu0hat <- as.numeric(predict(q0, newx = XD, s = "lambda.min"))
  mu1hat <- as.numeric(predict(q1, newx = XD, s = "lambda.min"))

  list(pihat = pihat, mu0hat = mu0hat, mu1hat = mu1hat)
}

fit_nuisance_cf_lasso <- function(Y, A, X, fold_ids, config) {
  n <- length(Y)
  bounds <- config$pi_bounds
  nf <- as.integer(config$cv_folds)
  XD <- .lasso_design(X)          # built once on full data -> aligned columns
  K <- max(fold_ids)

  pihat <- mu0hat <- mu1hat <- rep(NA_real_, n)
  for (k in 1:K) {
    tr <- which(fold_ids != k); te <- which(fold_ids == k)
    XDtr <- XD[tr, , drop = FALSE]; XDte <- XD[te, , drop = FALSE]
    Ytr <- Y[tr]; Atr <- A[tr]

    ps <- glmnet::cv.glmnet(XDtr, as.numeric(Atr), family = "binomial", alpha = 1, nfolds = nf)
    pihat[te] <- as.numeric(predict(ps, newx = XDte, s = "lambda.min", type = "response"))

    i0 <- Atr == 0; i1 <- Atr == 1
    q0 <- glmnet::cv.glmnet(XDtr[i0, , drop = FALSE], Ytr[i0], family = "gaussian", alpha = 1, nfolds = nf)
    q1 <- glmnet::cv.glmnet(XDtr[i1, , drop = FALSE], Ytr[i1], family = "gaussian", alpha = 1, nfolds = nf)
    mu0hat[te] <- as.numeric(predict(q0, newx = XDte, s = "lambda.min"))
    mu1hat[te] <- as.numeric(predict(q1, newx = XDte, s = "lambda.min"))
  }

  pihat <- pmax(bounds[1], pmin(bounds[2], pihat))
  list(pihat = pihat, mu0hat = mu0hat, mu1hat = mu1hat)
}

register_learner("lasso", fit_nuisance_lasso, fit_nuisance_cf_lasso,
                 packages = "glmnet")


# =============================================================================
# ==== Learner: knn
# =============================================================================

# =============================================================================
# knn — k-nearest-neighbors (caret::knnreg) for both nuisances
#
# Run on the 0/1 treatment, knnreg returns the treated fraction among the k
# nearest neighbors (an estimate of P(A=1|X)); run on Y it returns the neighbor
# mean. Predictors standardized (center/scale learned on training rows only).
# k CV-selected per nuisance from {5,10,15,20,30,50} minimizing CV MSE
# (= Brier score for the binary target) on SYSTEMATIC folds, so the whole
# learner is deterministic — no RNG.
#
# Known property: pihat is coarse (~k+1 distinct values), so kNN produces
# extreme inverse-propensity weights by construction.
# =============================================================================

suppressPackageStartupMessages(library(caret))

.KNN_K_GRID <- c(5L, 10L, 15L, 20L, 30L, 50L)

.knn_std_fit   <- function(M) {
  s <- apply(M, 2, sd); s[s == 0 | is.na(s)] <- 1
  list(center = colMeans(M), scale = s)
}
.knn_std_apply <- function(M, p) sweep(sweep(M, 2, p$center, "-"), 2, p$scale, "/")

.knn_best_k <- function(Xs, y, ks, nfolds) {
  n <- length(y)
  ks <- ks[ks >= 1 & ks < n]
  if (length(ks) == 0) return(max(1L, n - 1L))
  if (length(ks) == 1 || n < nfolds + 1) return(ks[1])
  folds <- rep_len(seq_len(nfolds), n)
  sse <- vapply(ks, function(k) {
    e <- 0
    for (f in seq_len(nfolds)) {
      tr <- folds != f; te <- !tr
      if (sum(tr) <= k) return(Inf)
      fit <- caret::knnreg(Xs[tr, , drop = FALSE], y[tr], k = k)
      e <- e + sum((y[te] - predict(fit, Xs[te, , drop = FALSE]))^2)
    }
    e
  }, numeric(1))
  ks[which.min(sse)]
}

.knn_fit <- function(ytr, Xtr, Xeval, ks, nfolds) {
  p <- .knn_std_fit(Xtr)
  Xs_tr <- .knn_std_apply(Xtr, p); Xs_ev <- .knn_std_apply(Xeval, p)
  k <- .knn_best_k(Xs_tr, ytr, ks, nfolds)
  fit <- caret::knnreg(Xs_tr, ytr, k = k)
  as.numeric(predict(fit, Xs_ev))
}

fit_nuisance_knn <- function(Y, A, X, config) {
  Xm <- as.matrix(as.data.frame(X))
  bounds <- config$pi_bounds
  ks <- .KNN_K_GRID; nf <- as.integer(config$cv_folds)

  pihat <- pmax(bounds[1], pmin(bounds[2],
            .knn_fit(as.numeric(A), Xm, Xm, ks, nf)))

  idx0 <- A == 0; idx1 <- A == 1
  mu0hat <- .knn_fit(Y[idx0], Xm[idx0, , drop = FALSE], Xm, ks, nf)
  mu1hat <- .knn_fit(Y[idx1], Xm[idx1, , drop = FALSE], Xm, ks, nf)

  list(pihat = pihat, mu0hat = mu0hat, mu1hat = mu1hat)
}

fit_nuisance_cf_knn <- function(Y, A, X, fold_ids, config) {
  n <- length(Y)
  Xm <- as.matrix(as.data.frame(X))
  bounds <- config$pi_bounds
  ks <- .KNN_K_GRID; nf <- as.integer(config$cv_folds)
  K <- max(fold_ids)

  pihat <- mu0hat <- mu1hat <- rep(NA_real_, n)
  for (k in 1:K) {
    tr <- which(fold_ids != k); te <- which(fold_ids == k)
    Xtr <- Xm[tr, , drop = FALSE]; Xte <- Xm[te, , drop = FALSE]
    Ytr <- Y[tr]; Atr <- A[tr]

    pihat[te] <- .knn_fit(as.numeric(Atr), Xtr, Xte, ks, nf)

    i0 <- Atr == 0; i1 <- Atr == 1
    mu0hat[te] <- .knn_fit(Ytr[i0], Xtr[i0, , drop = FALSE], Xte, ks, nf)
    mu1hat[te] <- .knn_fit(Ytr[i1], Xtr[i1, , drop = FALSE], Xte, ks, nf)
  }

  pihat <- pmax(bounds[1], pmin(bounds[2], pihat))
  list(pihat = pihat, mu0hat = mu0hat, mu1hat = mu1hat)
}

register_learner("knn", fit_nuisance_knn, fit_nuisance_cf_knn, packages = "caret")


# =============================================================================
# ==== Learner: bart
# =============================================================================

# =============================================================================
# bart — Bayesian Additive Regression Trees (dbarts::bart), Hill (2011)
#
# Propensity: probit BART on the 0/1 treatment, P(A=1|X) = posterior mean of
# Phi(f(x)) = colMeans(pnorm(yhat.test)). Outcome: Gaussian BART, arm-
# stratified, posterior-mean predictions. No standardization (trees are
# scale-invariant). Sampler: ntree = 200, ndpost = 500, nskip = 100, other
# priors at dbarts defaults (k = 2, power = 2, base = 0.95, numcut = 100).
# Content-seeded like mlp (MCMC is stochastic; dbarts honors R's set.seed).
# =============================================================================

suppressPackageStartupMessages(library(dbarts))

.BART_NTREE  <- 200L
.BART_NDPOST <- 500L
.BART_NSKIP  <- 100L

.bart_seed <- function(Y, A, X) {
  raw <- serialize(list(as.numeric(Y), as.numeric(A), as.numeric(as.matrix(X))),
                   connection = NULL, version = 2)
  as.integer(sum(as.numeric(raw)) %% 2147483647) + 1L
}

.bart_with_seed <- function(seed, expr) {
  if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
    old <- get(".Random.seed", envir = globalenv(), inherits = FALSE)
    on.exit(assign(".Random.seed", old, envir = globalenv()), add = TRUE)
  } else {
    on.exit(if (exists(".Random.seed", envir = globalenv(), inherits = FALSE))
              rm(".Random.seed", envir = globalenv()), add = TRUE)
  }
  set.seed(seed)
  expr
}

.bart_prob <- function(Atr, Xtr, Xeval) {
  if (length(unique(Atr)) < 2L) return(rep(mean(Atr), nrow(Xeval)))
  fit <- dbarts::bart(x.train = Xtr, y.train = Atr, x.test = Xeval,
                      ntree = .BART_NTREE, ndpost = .BART_NDPOST, nskip = .BART_NSKIP,
                      verbose = FALSE, keeptrees = FALSE)
  colMeans(pnorm(fit$yhat.test))
}

.bart_reg <- function(Ytr, Xtr, Xeval) {
  fit <- dbarts::bart(x.train = Xtr, y.train = Ytr, x.test = Xeval,
                      ntree = .BART_NTREE, ndpost = .BART_NDPOST, nskip = .BART_NSKIP,
                      verbose = FALSE, keeptrees = FALSE)
  as.numeric(fit$yhat.test.mean)
}

fit_nuisance_bart <- function(Y, A, X, config) {
  .bart_with_seed(.bart_seed(Y, A, X), {
    Xm <- as.matrix(as.data.frame(X))
    bounds <- config$pi_bounds

    pihat <- pmax(bounds[1], pmin(bounds[2],
              .bart_prob(as.numeric(A), Xm, Xm)))

    idx0 <- A == 0; idx1 <- A == 1
    mu0hat <- .bart_reg(Y[idx0], Xm[idx0, , drop = FALSE], Xm)
    mu1hat <- .bart_reg(Y[idx1], Xm[idx1, , drop = FALSE], Xm)

    list(pihat = pihat, mu0hat = mu0hat, mu1hat = mu1hat)
  })
}

fit_nuisance_cf_bart <- function(Y, A, X, fold_ids, config) {
  .bart_with_seed(.bart_seed(Y, A, X), {
    n <- length(Y)
    Xm <- as.matrix(as.data.frame(X))
    bounds <- config$pi_bounds
    K <- max(fold_ids)

    pihat <- mu0hat <- mu1hat <- rep(NA_real_, n)
    for (k in 1:K) {
      tr <- which(fold_ids != k); te <- which(fold_ids == k)
      Xtr <- Xm[tr, , drop = FALSE]; Xte <- Xm[te, , drop = FALSE]
      Ytr <- Y[tr]; Atr <- A[tr]

      pihat[te] <- .bart_prob(as.numeric(Atr), Xtr, Xte)

      i0 <- Atr == 0; i1 <- Atr == 1
      mu0hat[te] <- .bart_reg(Ytr[i0], Xtr[i0, , drop = FALSE], Xte)
      mu1hat[te] <- .bart_reg(Ytr[i1], Xtr[i1, , drop = FALSE], Xte)
    }

    pihat <- pmax(bounds[1], pmin(bounds[2], pihat))
    list(pihat = pihat, mu0hat = mu0hat, mu1hat = mu1hat)
  })
}

register_learner("bart", fit_nuisance_bart, fit_nuisance_cf_bart, packages = "dbarts")


# =============================================================================
# ==== Learner: ranger
# =============================================================================

# =============================================================================
# ranger — random forest: 500 trees, mtry = 2, min.node.size CV-selected from
# {30, 60} per nuisance (V = config$cv_folds). Propensity: probability forest;
# outcome: regression forests, arm-stratified. Stochastic (bootstrap + CV
# splits); seeded upstream by the dispatcher.
# =============================================================================

suppressPackageStartupMessages(library(ranger))

.select_ranger_node_size <- function(X, Y, family, cv_folds = 5,
                                     candidates = c(30, 60)) {
  n <- length(Y)
  X_df <- as.data.frame(X)
  fids <- sample(rep(1:cv_folds, ceiling(n / cv_folds))[1:n])

  cv_errors <- numeric(length(candidates))
  for (c_idx in seq_along(candidates)) {
    node_size <- candidates[c_idx]
    fold_errors <- numeric(cv_folds)
    for (v in 1:cv_folds) {
      val_idx <- which(fids == v)
      train_idx <- which(fids != v)
      if (family == "binomial") {
        fit <- ranger(y = factor(Y[train_idx]),
                      x = X_df[train_idx, , drop = FALSE],
                      num.trees = 500, mtry = 2,
                      min.node.size = node_size, probability = TRUE)
        pred <- predict(fit, data = X_df[val_idx, , drop = FALSE])$predictions[, 2]
      } else {
        fit <- ranger(y = Y[train_idx],
                      x = X_df[train_idx, , drop = FALSE],
                      num.trees = 500, mtry = 2,
                      min.node.size = node_size)
        pred <- predict(fit, data = X_df[val_idx, , drop = FALSE])$predictions
      }
      fold_errors[v] <- mean((Y[val_idx] - pred)^2)
    }
    cv_errors[c_idx] <- mean(fold_errors)
  }
  candidates[which.min(cv_errors)]
}

fit_nuisance_ranger <- function(Y, A, X, config) {
  X_df <- as.data.frame(X)
  bounds <- config$pi_bounds
  cv_folds <- config$cv_folds

  ps_node <- .select_ranger_node_size(X_df, as.numeric(A), "binomial", cv_folds)
  g_fit <- ranger(y = factor(A), x = X_df, num.trees = 500, mtry = 2,
                  min.node.size = ps_node, probability = TRUE)
  pihat <- pmax(bounds[1], pmin(bounds[2],
                predict(g_fit, data = X_df)$predictions[, 2]))

  idx_0 <- A == 0
  Q0_node <- .select_ranger_node_size(X_df[idx_0, , drop = FALSE], Y[idx_0], "gaussian", cv_folds)
  Q0_fit <- ranger(y = Y[idx_0], x = X_df[idx_0, , drop = FALSE],
                   num.trees = 500, mtry = 2, min.node.size = Q0_node)
  mu0hat <- predict(Q0_fit, data = X_df)$predictions

  idx_1 <- A == 1
  Q1_node <- .select_ranger_node_size(X_df[idx_1, , drop = FALSE], Y[idx_1], "gaussian", cv_folds)
  Q1_fit <- ranger(y = Y[idx_1], x = X_df[idx_1, , drop = FALSE],
                   num.trees = 500, mtry = 2, min.node.size = Q1_node)
  mu1hat <- predict(Q1_fit, data = X_df)$predictions

  list(pihat = pihat, mu0hat = mu0hat, mu1hat = mu1hat)
}

fit_nuisance_cf_ranger <- function(Y, A, X, fold_ids, config) {
  n <- length(Y)
  X_df <- as.data.frame(X)
  bounds <- config$pi_bounds
  cv_folds <- config$cv_folds
  K <- max(fold_ids)

  pihat <- mu0hat <- mu1hat <- rep(NA_real_, n)
  for (k in 1:K) {
    train_idx <- which(fold_ids != k)
    test_idx  <- which(fold_ids == k)
    X_train <- X_df[train_idx, , drop = FALSE]
    X_test  <- X_df[test_idx,  , drop = FALSE]
    A_train <- A[train_idx]
    Y_train <- Y[train_idx]

    ps_node <- .select_ranger_node_size(X_train, as.numeric(A_train), "binomial", cv_folds)
    g_fit <- ranger(y = factor(A_train), x = X_train, num.trees = 500, mtry = 2,
                    min.node.size = ps_node, probability = TRUE)
    pihat[test_idx] <- predict(g_fit, data = X_test)$predictions[, 2]

    t0 <- which(A_train == 0)
    Q0_node <- .select_ranger_node_size(X_train[t0, , drop = FALSE],
                                        Y_train[t0], "gaussian", cv_folds)
    Q0_fit <- ranger(y = Y_train[t0], x = X_train[t0, , drop = FALSE],
                     num.trees = 500, mtry = 2, min.node.size = Q0_node)
    mu0hat[test_idx] <- predict(Q0_fit, data = X_test)$predictions

    t1 <- which(A_train == 1)
    Q1_node <- .select_ranger_node_size(X_train[t1, , drop = FALSE],
                                        Y_train[t1], "gaussian", cv_folds)
    Q1_fit <- ranger(y = Y_train[t1], x = X_train[t1, , drop = FALSE],
                     num.trees = 500, mtry = 2, min.node.size = Q1_node)
    mu1hat[test_idx] <- predict(Q1_fit, data = X_test)$predictions
  }

  pihat <- pmax(bounds[1], pmin(bounds[2], pihat))
  list(pihat = pihat, mu0hat = mu0hat, mu1hat = mu1hat)
}

register_learner("ranger", fit_nuisance_ranger, fit_nuisance_cf_ranger,
                 packages = "ranger")


# =============================================================================
# ==== Learner: xgboost_naimi
# =============================================================================

# =============================================================================
# xgboost_naimi — gradient boosting at the Naimi et al. (2023) specification:
# 500 rounds, max_depth = 4, eta = 0.1, min_child_weight CV-selected from
# {30, 60} per nuisance (nfold = config$cv_folds). Arm-stratified outcome.
# Stochastic (CV fold assignment); seeded upstream by the dispatcher.
# =============================================================================

suppressPackageStartupMessages(library(xgboost))

.xgb_select_node_size <- function(X_mat, Y_vec, objective, cv_folds = 5,
                                  candidates = c(30, 60)) {
  eval_metric <- if (objective == "binary:logistic") "logloss" else "rmse"
  dtrain <- xgb.DMatrix(data = X_mat, label = Y_vec)

  cv_scores <- numeric(length(candidates))
  for (c_idx in seq_along(candidates)) {
    cv_result <- xgb.cv(
      params = list(objective = objective, eval_metric = eval_metric,
                    max_depth = 4, eta = 0.1,
                    min_child_weight = candidates[c_idx]),
      data = dtrain, nrounds = 500, nfold = cv_folds, verbose = 0)
    eval_col <- paste0("test_", eval_metric, "_mean")
    cv_scores[c_idx] <- min(cv_result$evaluation_log[[eval_col]])
  }
  candidates[which.min(cv_scores)]
}

fit_nuisance_xgboost_naimi <- function(Y, A, X, config) {
  X_mat <- as.matrix(X)
  bounds <- config$pi_bounds
  cv_folds <- config$cv_folds

  ps_node <- .xgb_select_node_size(X_mat, as.numeric(A), "binary:logistic", cv_folds)
  ps_fit <- xgb.train(
    params = list(objective = "binary:logistic", eval_metric = "logloss",
                  max_depth = 4, eta = 0.1, min_child_weight = ps_node),
    data = xgb.DMatrix(data = X_mat, label = as.numeric(A)),
    nrounds = 500, verbose = 0)
  pihat <- pmax(bounds[1], pmin(bounds[2], predict(ps_fit, X_mat)))

  idx_0 <- A == 0
  Q0_node <- .xgb_select_node_size(X_mat[idx_0, , drop = FALSE], Y[idx_0],
                                   "reg:squarederror", cv_folds)
  Q0_fit <- xgb.train(
    params = list(objective = "reg:squarederror", eval_metric = "rmse",
                  max_depth = 4, eta = 0.1, min_child_weight = Q0_node),
    data = xgb.DMatrix(data = X_mat[idx_0, , drop = FALSE], label = Y[idx_0]),
    nrounds = 500, verbose = 0)
  mu0hat <- predict(Q0_fit, X_mat)

  idx_1 <- A == 1
  Q1_node <- .xgb_select_node_size(X_mat[idx_1, , drop = FALSE], Y[idx_1],
                                   "reg:squarederror", cv_folds)
  Q1_fit <- xgb.train(
    params = list(objective = "reg:squarederror", eval_metric = "rmse",
                  max_depth = 4, eta = 0.1, min_child_weight = Q1_node),
    data = xgb.DMatrix(data = X_mat[idx_1, , drop = FALSE], label = Y[idx_1]),
    nrounds = 500, verbose = 0)
  mu1hat <- predict(Q1_fit, X_mat)

  list(pihat = pihat, mu0hat = mu0hat, mu1hat = mu1hat)
}

fit_nuisance_cf_xgboost_naimi <- function(Y, A, X, fold_ids, config) {
  n <- length(Y)
  X_mat <- as.matrix(X)
  bounds <- config$pi_bounds
  cv_folds <- config$cv_folds
  K <- max(fold_ids)

  pihat <- mu0hat <- mu1hat <- rep(NA_real_, n)

  fit_one <- function(X_tr, Y_tr, objective) {
    node <- .xgb_select_node_size(X_tr, Y_tr, objective, cv_folds)
    dtrain <- xgb.DMatrix(data = X_tr, label = Y_tr)
    eval_metric <- if (objective == "binary:logistic") "logloss" else "rmse"
    xgb.train(
      params = list(objective = objective, eval_metric = eval_metric,
                    max_depth = 4, eta = 0.1, min_child_weight = node),
      data = dtrain, nrounds = 500, verbose = 0)
  }

  for (k in 1:K) {
    train_idx <- which(fold_ids != k)
    test_idx  <- which(fold_ids == k)
    X_train <- X_mat[train_idx, , drop = FALSE]
    X_test  <- X_mat[test_idx,  , drop = FALSE]
    Y_train <- Y[train_idx]
    A_train <- A[train_idx]

    ps_fit <- fit_one(X_train, as.numeric(A_train), "binary:logistic")
    pihat[test_idx] <- predict(ps_fit, X_test)

    i0 <- which(A_train == 0)
    Q0_fit <- fit_one(X_train[i0, , drop = FALSE], Y_train[i0], "reg:squarederror")
    mu0hat[test_idx] <- predict(Q0_fit, X_test)

    i1 <- which(A_train == 1)
    Q1_fit <- fit_one(X_train[i1, , drop = FALSE], Y_train[i1], "reg:squarederror")
    mu1hat[test_idx] <- predict(Q1_fit, X_test)
  }

  pihat <- pmax(bounds[1], pmin(bounds[2], pihat))
  list(pihat = pihat, mu0hat = mu0hat, mu1hat = mu1hat)
}

register_learner("xgboost_naimi", fit_nuisance_xgboost_naimi,
                 fit_nuisance_cf_xgboost_naimi, packages = "xgboost")


# =============================================================================
# ==== Learner: hal_discrete
# =============================================================================

# =============================================================================
# hal_discrete — discrete SuperLearner over two Highly Adaptive Lasso fits
# (hal9001::fit_hal, max_degree = 2, smoothness_orders 0 vs 1, no basis
# reduction), selecting per nuisance by the lasso's cross-validated risk.
# Propensity: binomial; outcome: gaussian, arm-stratified. hal9001's internal
# lambda CV uses its own default 10-fold split (package hardcoded).
#
# CF runs the K folds in parallel via mclapply. Unlike the production
# implementation — whose fold workers were re-seeded from time+PID by
# mclapply's default (mc.set.seed=TRUE under Mersenne-Twister), making
# archived hal_discrete CF rows non-reproducible (see DEVIATIONS D8) — this
# version switches to L'Ecuyer-CMRG so each fold job gets its own
# DETERMINISTIC stream derived from the ambient (fold-seed-pinned) state.
# Results are identical across runs and across HAL_CF_CORES settings.
# Cap forks with env var HAL_CF_CORES (default: physical cores - 1).
# =============================================================================

suppressPackageStartupMessages({ library(hal9001); library(parallel) })

.fit_hal_discrete <- function(X_mat, Y_vec, family, reduce_basis, bounds = NULL) {
  configs <- list(
    list(smoothness_orders = 0, label = "HAL(s=0,d=2)"),
    list(smoothness_orders = 1, label = "HAL(s=1,d=2)")
  )

  best_risk <- Inf
  best_fit <- NULL

  for (cfg in configs) {
    fit <- tryCatch(
      fit_hal(X = X_mat, Y = Y_vec, family = family, max_degree = 2,
              smoothness_orders = cfg$smoothness_orders,
              reduce_basis = reduce_basis),
      error = function(e) {
        warning(sprintf("HAL fit failed (%s): %s", cfg$label, e$message))
        NULL
      })
    if (is.null(fit)) next
    cv_risk <- min(fit$lasso_fit$cvm)
    if (cv_risk < best_risk) {
      best_risk <- cv_risk
      best_fit <- fit
    }
  }

  if (is.null(best_fit)) stop("Both HAL fits failed")
  list(fit = best_fit)
}

fit_nuisance_hal_discrete <- function(Y, A, X, config) {
  X_mat <- as.matrix(X)
  bounds <- config$pi_bounds
  rb <- NULL

  ps_result <- .fit_hal_discrete(X_mat, as.numeric(A), "binomial", rb, bounds)
  pihat <- as.numeric(predict(ps_result$fit, new_data = X_mat, type = "response"))
  pihat <- pmax(bounds[1], pmin(bounds[2], pihat))

  idx_0 <- A == 0
  Q0_result <- .fit_hal_discrete(X_mat[idx_0, , drop = FALSE], Y[idx_0], "gaussian", rb)
  mu0hat <- as.numeric(predict(Q0_result$fit, new_data = X_mat))

  idx_1 <- A == 1
  Q1_result <- .fit_hal_discrete(X_mat[idx_1, , drop = FALSE], Y[idx_1], "gaussian", rb)
  mu1hat <- as.numeric(predict(Q1_result$fit, new_data = X_mat))

  list(pihat = pihat, mu0hat = mu0hat, mu1hat = mu1hat)
}

.fit_one_fold_hal_discrete <- function(k, Y, A, X_mat, fold_ids, bounds, rb) {
  t_fold_start <- Sys.time()
  train <- fold_ids != k
  test  <- fold_ids == k
  X_tr  <- X_mat[train, , drop = FALSE]
  X_te  <- X_mat[test,  , drop = FALSE]
  Y_tr  <- Y[train]
  A_tr  <- A[train]

  ps_res <- .fit_hal_discrete(X_tr, as.numeric(A_tr), "binomial", rb, bounds)
  pihat_k <- as.numeric(predict(ps_res$fit, new_data = X_te, type = "response"))
  pihat_k <- pmax(bounds[1], pmin(bounds[2], pihat_k))

  i0 <- which(A_tr == 0)
  Q0_res <- .fit_hal_discrete(X_tr[i0, , drop = FALSE], Y_tr[i0], "gaussian", rb)
  mu0_k  <- as.numeric(predict(Q0_res$fit, new_data = X_te))

  i1 <- which(A_tr == 1)
  Q1_res <- .fit_hal_discrete(X_tr[i1, , drop = FALSE], Y_tr[i1], "gaussian", rb)
  mu1_k  <- as.numeric(predict(Q1_res$fit, new_data = X_te))

  list(test = which(test), pihat_k = pihat_k, mu0_k = mu0_k, mu1_k = mu1_k,
       wall_sec = as.numeric(difftime(Sys.time(), t_fold_start, units = "secs")))
}

fit_nuisance_cf_hal_discrete <- function(Y, A, X, fold_ids, config) {
  X_mat  <- as.matrix(X)
  bounds <- config$pi_bounds
  n      <- length(Y)
  K      <- max(fold_ids)
  rb     <- NULL

  cores_cap <- suppressWarnings(as.integer(Sys.getenv("HAL_CF_CORES", "")))
  if (is.na(cores_cap) || cores_cap < 1L) {
    cores_cap <- max(1L, parallel::detectCores(logical = FALSE) - 1L)
  }
  mc_cores <- min(K, cores_cap)

  # Deterministic per-fold RNG streams: seed drawn from the ambient stream
  # (pinned upstream by the CF fold seed), then L'Ecuyer-CMRG + mc.set.seed
  # assigns each fold job its own reproducible stream in job order. The
  # caller's RNG kind and state are restored on exit (.Random.seed encodes
  # the kind, so restoring it restores both).
  stream_seed <- sample.int(.Machine$integer.max, 1L)
  old_seed <- get(".Random.seed", envir = globalenv(), inherits = FALSE)
  on.exit(assign(".Random.seed", old_seed, envir = globalenv()), add = TRUE)
  RNGkind("L'Ecuyer-CMRG")
  set.seed(stream_seed)

  fold_results <- parallel::mclapply(
    seq_len(K), .fit_one_fold_hal_discrete,
    Y = Y, A = A, X_mat = X_mat, fold_ids = fold_ids, bounds = bounds, rb = rb,
    mc.cores = mc_cores, mc.preschedule = FALSE, mc.set.seed = TRUE)

  errs <- vapply(fold_results, inherits, logical(1), what = "try-error")
  if (any(errs)) {
    err_idx <- which(errs)[1]
    stop(sprintf("hal_discrete CF fold %d failed: %s",
                 err_idx, attr(fold_results[[err_idx]], "condition")$message))
  }

  pihat <- mu0hat <- mu1hat <- numeric(n)
  for (fr in fold_results) {
    pihat[fr$test]  <- fr$pihat_k
    mu0hat[fr$test] <- fr$mu0_k
    mu1hat[fr$test] <- fr$mu1_k
  }

  list(pihat = pihat, mu0hat = mu0hat, mu1hat = mu1hat)
}

register_learner("hal_discrete", fit_nuisance_hal_discrete,
                 fit_nuisance_cf_hal_discrete,
                 packages = c("hal9001", "parallel"))


# =============================================================================
# ==== Learner: sl_default
# =============================================================================

# =============================================================================
# sl_default — the tmle package's default SuperLearner libraries
#   propensity: SL.glm + tmle.SL.dbarts.k.5 + SL.gam
#   outcome:    SL.glm + tmle.SL.dbarts2    + SL.glmnet
# =============================================================================

suppressPackageStartupMessages(library(tmle))   # provides tmle.SL.dbarts* wrappers

.SL_DEFAULT_PS  <- c("SL.glm", "tmle.SL.dbarts.k.5", "SL.gam")
.SL_DEFAULT_OUT <- c("SL.glm", "tmle.SL.dbarts2", "SL.glmnet")

register_learner(
  "sl_default",
  function(Y, A, X, config)
    sl_fit_nuisance(Y, A, X, config, .SL_DEFAULT_PS, .SL_DEFAULT_OUT),
  function(Y, A, X, fold_ids, config)
    sl_fit_nuisance_cf(Y, A, X, fold_ids, config, .SL_DEFAULT_PS, .SL_DEFAULT_OUT),
  packages = c("SuperLearner", "tmle", "dbarts", "gam", "glmnet"))


# =============================================================================
# ==== Learner: sl_naimi_v2
# =============================================================================

# =============================================================================
# sl_naimi_v2 — the sl_naimi_v1 library fit on covariates augmented with all
# pairwise interaction features.
# =============================================================================


register_learner(
  "sl_naimi_v2",
  function(Y, A, X, config)
    sl_fit_nuisance(Y, A, X, config, create_naimi_library(),
                    transform_X = add_pairwise_interactions),
  function(Y, A, X, fold_ids, config)
    sl_fit_nuisance_cf(Y, A, X, fold_ids, config, create_naimi_library(),
                       transform_X = add_pairwise_interactions),
  packages = c("SuperLearner", "ranger", "xgboost", "gam"))


# =============================================================================
# ==== Learner: sl_balzer
# =============================================================================

# =============================================================================
# sl_balzer — Balzer-style SuperLearner library:
# SL.glm + SL.step.interaction + SL.earth (MARS) + SL.mean
# =============================================================================


.SL_BALZER <- c("SL.glm", "SL.step.interaction", "SL.earth", "SL.mean")

register_learner(
  "sl_balzer",
  function(Y, A, X, config)
    sl_fit_nuisance(Y, A, X, config, .SL_BALZER),
  function(Y, A, X, fold_ids, config)
    sl_fit_nuisance_cf(Y, A, X, fold_ids, config, .SL_BALZER),
  packages = c("SuperLearner", "earth"))


# =============================================================================
# ==== Learner: hal_n5000 (simplified HAL, the study's HAL at n = 5000)
# =============================================================================

# =============================================================================
# hal_n5000 — simplified single-fit HAL, the study's "HAL" at n = 5000
#
# hal_discrete (the n=200/n=1200 HAL) fits both smoothness-order candidates
# (s=0, s=1) per nuisance and lets a discrete SL pick by CV risk. At n=5000
# that costs ~14 min per nuisance fit, so the study ran this deliberately
# simplified variant instead: ONE fit_hal() per nuisance at max_degree = 2,
# smoothness_orders = 1 (the candidate hal_discrete selects on this DGP),
# internal lambda-selection CV = config$cv_folds (5 at n=5000), no
# reduce_basis. Outcome models are arm-stratified like every other learner.
#
# Cross-fitting runs the folds in parallel (mclapply; HAL_CF_CORES caps the
# fork count) with a deterministic per-fold seed set INSIDE each fork
# (sim_seed*1000 + 70 + k), so results are reproducible and independent of
# the core count.
# =============================================================================

library(hal9001)

.hal5000_fit_predict <- function(X_tr, Y_tr, family, X_pred, nfolds) {
  fit <- tryCatch(
    fit_hal(X = X_tr, Y = Y_tr, family = family,
            max_degree = 2, smoothness_orders = 1,
            fit_control = list(nfolds = nfolds)),
    error = function(e) {
      warning(paste("HAL fit failed:", e$message, "- using GLM"))
      glm(Y ~ ., data = data.frame(Y = Y_tr, as.data.frame(X_tr)),
          family = if (family == "binomial") binomial() else gaussian())
    }
  )
  if (inherits(fit, "hal9001")) {
    if (family == "binomial")
      as.numeric(predict(fit, new_data = X_pred, type = "response"))
    else
      as.numeric(predict(fit, new_data = X_pred))
  } else {
    if (family == "binomial")
      as.numeric(predict(fit, newdata = as.data.frame(X_pred), type = "response"))
    else
      as.numeric(predict(fit, newdata = as.data.frame(X_pred)))
  }
}

fit_nuisance_hal_n5000 <- function(Y, A, X, config) {
  X_mat  <- as.matrix(X)
  bounds <- config$pi_bounds
  nfolds <- as.integer(config$cv_folds)

  pihat <- .hal5000_fit_predict(X_mat, as.numeric(A), "binomial", X_mat, nfolds)
  pihat <- pmax(bounds[1], pmin(bounds[2], pihat))

  idx_0 <- A == 0; idx_1 <- A == 1
  mu0hat <- .hal5000_fit_predict(X_mat[idx_0, , drop = FALSE], as.numeric(Y[idx_0]),
                                 "gaussian", X_mat, nfolds)
  mu1hat <- .hal5000_fit_predict(X_mat[idx_1, , drop = FALSE], as.numeric(Y[idx_1]),
                                 "gaussian", X_mat, nfolds)

  list(pihat = pihat, mu0hat = mu0hat, mu1hat = mu1hat)
}

.hal5000_fit_one_fold <- function(k, Y, A, X_mat, fold_ids, bounds, nfolds,
                                  base_seed = NULL) {
  # Pin this fork's RNG so the inner lambda-selection CV is deterministic
  # regardless of mc.cores. Distinct per fold (k) and per sim (base_seed).
  if (!is.null(base_seed)) set.seed(base_seed * 1000 + 70 + k)
  train <- fold_ids != k
  test  <- fold_ids == k
  X_tr  <- X_mat[train, , drop = FALSE]
  X_te  <- X_mat[test,  , drop = FALSE]
  Y_tr  <- Y[train]; A_tr <- A[train]

  pihat_k <- .hal5000_fit_predict(X_tr, as.numeric(A_tr), "binomial", X_te, nfolds)
  pihat_k <- pmax(bounds[1], pmin(bounds[2], pihat_k))

  i0 <- which(A_tr == 0)
  mu0_k <- .hal5000_fit_predict(X_tr[i0, , drop = FALSE], as.numeric(Y_tr[i0]),
                                "gaussian", X_te, nfolds)
  i1 <- which(A_tr == 1)
  mu1_k <- .hal5000_fit_predict(X_tr[i1, , drop = FALSE], as.numeric(Y_tr[i1]),
                                "gaussian", X_te, nfolds)

  list(test = which(test), pihat_k = pihat_k, mu0_k = mu0_k, mu1_k = mu1_k)
}

fit_nuisance_cf_hal_n5000 <- function(Y, A, X, fold_ids, config) {
  X_mat  <- as.matrix(X)
  bounds <- config$pi_bounds
  n      <- length(Y)
  K      <- max(fold_ids)
  nfolds <- as.integer(config$cv_folds)
  base_seed <- config$sim_seed

  cores_cap <- suppressWarnings(as.integer(Sys.getenv("HAL_CF_CORES", "")))
  if (is.na(cores_cap) || cores_cap < 1L) {
    cores_cap <- max(1L, parallel::detectCores(logical = FALSE) - 1L)
  }
  mc_cores <- min(K, cores_cap)

  fold_results <- parallel::mclapply(
    seq_len(K), .hal5000_fit_one_fold,
    Y = Y, A = A, X_mat = X_mat, fold_ids = fold_ids, bounds = bounds,
    nfolds = nfolds, base_seed = base_seed,
    mc.cores = mc_cores, mc.preschedule = FALSE
  )

  errs <- vapply(fold_results, inherits, logical(1), what = "try-error")
  if (any(errs)) {
    ei <- which(errs)[1]
    stop(sprintf("hal_n5000 CF fold %d failed: %s", ei,
                 attr(fold_results[[ei]], "condition")$message))
  }

  pihat <- numeric(n); mu0hat <- numeric(n); mu1hat <- numeric(n)
  for (fr in fold_results) {
    pihat[fr$test]  <- fr$pihat_k
    mu0hat[fr$test] <- fr$mu0_k
    mu1hat[fr$test] <- fr$mu1_k
  }

  list(pihat = pihat, mu0hat = mu0hat, mu1hat = mu1hat)
}

register_learner("hal_n5000", fit_nuisance_hal_n5000, fit_nuisance_cf_hal_n5000,
                 packages = "hal9001")


# =============================================================================
# ==== The four estimators (IPW / G-computation / AIPW / TMLE)
# =============================================================================

# =============================================================================
# R/estimators.R — the four ATE estimators
#
# All four consume the same nuisance list(pihat, mu0hat, mu1hat) produced by
# one learner, so within a simulation the estimator contrast is exactly a
# contrast of estimation strategies on identical inputs.
#
#   ipw    stabilized Hajek weights in a weighted OLS of Y ~ A;
#          HC sandwich SE (sandwich::vcovHC type "HC")
#   gcomp  plug-in mean(mu1hat - mu0hat); NO standard error by design
#          (its rows are excluded from coverage summaries)
#   aipw   one-step efficient-influence-function estimator; SE = sd(psi)/sqrt(n)
#   tmle   tmle::tmle() with Q and g1W supplied (logistic fluctuation on
#          scaled Y); gbound is passed EXPLICITLY as the study's pi_bounds.
#          Without it tmle() silently applies its own adaptive floor
#          5/(sqrt(n)*ln n) to each arm's weight denominator — at n=200 that
#          floor (0.0667) is WIDER than the study truncation [0.025, 0.975]
#          and would re-truncate the propensities. Passing gbound = pi_bounds
#          makes the truncation policy identical at every sample size.
#
# Wald 95% CIs (estimate ± 1.96·SE) except tmle, which reports its own IC CI.
# =============================================================================

suppressPackageStartupMessages({
  library(sandwich)
  library(tmle)
})

estimate_ipw <- function(Y, A, X, nuisance) {
  pihat <- nuisance$pihat
  weights <- A * (mean(A) / pihat) + (1 - A) * ((1 - mean(A)) / (1 - pihat))
  model <- lm(Y ~ A, weights = weights)
  estimate <- unname(coef(model)[2])
  se <- sqrt(vcovHC(model, type = "HC")[2, 2])
  list(estimate = estimate, se = se,
       ci_lower = estimate - 1.96 * se, ci_upper = estimate + 1.96 * se)
}

estimate_gcomp <- function(Y, A, X, nuisance) {
  list(estimate = mean(nuisance$mu1hat - nuisance$mu0hat),
       se = NA_real_, ci_lower = NA_real_, ci_upper = NA_real_)
}

estimate_aipw <- function(Y, A, X, nuisance) {
  n <- length(Y)
  pihat  <- nuisance$pihat
  mu0hat <- nuisance$mu0hat
  mu1hat <- nuisance$mu1hat
  muhat  <- A * mu1hat + (1 - A) * mu0hat
  psi <- ((2 * A - 1) * (Y - muhat)) / ((2 * A - 1) * pihat + (1 - A)) +
         mu1hat - mu0hat
  estimate <- mean(psi)
  se <- sd(psi) / sqrt(n)
  list(estimate = estimate, se = se,
       ci_lower = estimate - 1.96 * se, ci_upper = estimate + 1.96 * se)
}

estimate_tmle <- function(Y, A, X, nuisance) {
  gb <- nuisance$pi_bounds
  if (is.null(gb)) gb <- c(0.025, 0.975)
  # evalATT = FALSE: only estimates$ATE is recorded, but tmle() computes ATT/ATC
  # by default and may REFIT g by SuperLearner for them (default library incl.
  # dbarts, V = 10) whenever in-sample pihat separates the arms — the memorizing
  # NoCF learners — costing 10-20 s per cell. Verified 2026-08-10: psi/var/both
  # CI bounds are BIT-IDENTICAL with and without it, so this is a pure speedup
  # that changes no stored result.
  result <- tmle(Y = Y, A = A, W = as.data.frame(X),
                 Q = cbind(nuisance$mu0hat, nuisance$mu1hat),
                 g1W = nuisance$pihat,
                 gbound = gb,
                 evalATT = FALSE)
  ate <- result$estimates$ATE
  list(estimate = ate$psi, se = sqrt(ate$var.psi),
       ci_lower = ate$CI[1], ci_upper = ate$CI[2])
}

ESTIMATORS <- list(ipw   = estimate_ipw,
                   gcomp = estimate_gcomp,
                   aipw  = estimate_aipw,
                   tmle  = estimate_tmle)

#' Run one estimator with timing; errors become NA rows, not crashes
run_estimator <- function(name, Y, A, X, nuisance) {
  pt <- proc.time()
  result <- tryCatch(
    ESTIMATORS[[name]](Y = Y, A = A, X = X, nuisance = nuisance),
    error = function(e) list(estimate = NA_real_, se = NA_real_,
                             ci_lower = NA_real_, ci_upper = NA_real_,
                             error = conditionMessage(e))
  )
  el <- proc.time() - pt
  result$runtime      <- el[["user.self"]] + el[["sys.self"]]
  result$runtime_wall <- el[["elapsed"]]
  result
}


# =============================================================================
# ==== Driver: R-learner track
# =============================================================================

# --- R-learner track: fit nuisances + run all four estimators ----------------
# For each (scenario, learner, cross_fit, sim):
#   1. load the cached dataset (validated against the config)
#   2. estimate (pihat, mu0hat, mu1hat) via the learner (seeded; see the RNG
#      rules section)
#   3. run IPW / G-comp / AIPW / TMLE on those same nuisance vectors
#   4. append one row per estimator to the per-config results CSV
#
# Results land in results/n<k>/per_config/<scenario>_<learner>_<cf>_n<k>.csv.
# Runs are resumable: a sim counts as complete only if every estimator row is
# present with a non-NA estimate; anything else is re-queued. An unreadable
# results file is a hard error (resuming blindly would append duplicates).
#
# The oracle reference arm (config reference_learners) is handled here
# directly: its "nuisances" are the DGP truth stored in the cache
# (pi_true, mu0_true, mu1_true), which fitted learners never see, with the
# true pi truncated to pi_bounds (apples-to-apples ceiling). Cross-fitting is
# a no-op for ground truth, so both CF labels are written from a single
# computation per sim.

n         <- CONFIG$resolved$n
true_ate  <- CONFIG$dgp$true_ate
data_dir  <- CONFIG$resolved$data_dir
out_dir   <- file.path(CONFIG$resolved$results_dir, "per_config")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

est_config <- list(pi_bounds = as.numeric(unlist(CONFIG$estimation$pi_bounds)),
                   n_splits  = CONFIG$resolved$n_splits,
                   cv_folds  = CONFIG$resolved$cv_folds)
estimator_names <- unlist(CONFIG$estimators)

RESULT_COLS <- c("sim_id", "seed", "n", "scenario", "learner", "cross_fit",
                 "estimator", "estimate", "se", "ci_lower", "ci_upper",
                 "runtime", "nuisance_time", "nuisance_time_wall",
                 "true_ate", "bias", "covered")

fitted_learners <- intersect(OPTS$learners, names(LEARNERS))
unknown <- setdiff(OPTS$learners,
                   c(names(LEARNERS), unlist(CONFIG$reference_learners)))
if (length(unknown) > 0)
  stop(sprintf("Unknown learner(s): %s", paste(unknown, collapse = ", ")))
oracle_learners <- intersect(OPTS$learners, unlist(CONFIG$reference_learners))
# When invoked without --learners, run the oracle arm too.
if (identical(sort(OPTS$learners), sort(default_roster)))
  oracle_learners <- unlist(CONFIG$reference_learners)

read_sim <- function(sim_id) {
  f <- file.path(data_dir, sprintf("sim_%04d.rds", sim_id))
  if (!file.exists(f)) stop(sprintf("Missing dataset %s — run script 01 first", f))
  s <- readRDS(f)
  if (!isTRUE(s$n == n) || !isTRUE(s$true_ate == true_ate))
    stop(sprintf("Cached %s does not match config (n=%s, true_ate=%s)", f, s$n, s$true_ate))
  s
}

result_row <- function(sim_id, seed, scenario, learner, cross_fit, est_name,
                       r, nuisance) {
  bias <- r$estimate - true_ate
  data.frame(
    sim_id = sim_id, seed = seed, n = n, scenario = scenario,
    learner = learner, cross_fit = cross_fit, estimator = est_name,
    estimate = r$estimate, se = r$se, ci_lower = r$ci_lower, ci_upper = r$ci_upper,
    runtime = r$runtime,
    nuisance_time = nuisance$nuisance_time %||% NA_real_,
    nuisance_time_wall = nuisance$nuisance_time_wall %||% NA_real_,
    true_ate = true_ate, bias = bias,
    covered = if (!is.na(r$ci_lower) && !is.na(r$ci_upper))
                (r$ci_lower <= true_ate) & (true_ate <= r$ci_upper) else NA,
    stringsAsFactors = FALSE)
}

`%||%` <- function(a, b) if (is.null(a)) b else a

# Resume helper: which sims in this config file are already complete?
completed_sims <- function(output_file) {
  if (!file.exists(output_file)) return(integer(0))
  existing <- tryCatch(read.csv(output_file, stringsAsFactors = FALSE),
                       error = function(e)
                         stop(sprintf("Unreadable results file %s (%s); inspect or remove it.",
                                      output_file, conditionMessage(e))))
  if (nrow(existing) == 0) return(integer(0))
  ok <- vapply(split(existing, existing$sim_id),
               function(d) all(estimator_names %in% d$estimator) && !any(is.na(d$estimate)),
               logical(1))
  bad <- as.integer(names(ok)[!ok])
  if (length(bad) > 0) {
    message(sprintf("    re-queueing %d failed sims", length(bad)))
    write.csv(existing[!existing$sim_id %in% bad, , drop = FALSE], output_file, row.names = FALSE)
  }
  as.integer(names(ok)[ok])
}

append_rows <- function(df, output_file) {
  df <- df[, RESULT_COLS]
  write.table(df, output_file, sep = ",", row.names = FALSE,
              col.names = !file.exists(output_file), append = file.exists(output_file))
}

if (OPTS$track %in% c("r", "all")) {

# --- Fitted learners ---------------------------------------------------------
for (scenario in OPTS$scenarios) {
  for (learner_name in fitted_learners) {
    for (cross_fit in OPTS$cross_fit) {
      cf_label <- if (cross_fit) "cf" else "nocf"
      output_file <- file.path(out_dir, sprintf("%s_%s_%s_n%d.csv",
                                                scenario, learner_name, cf_label, n))
      if (cross_fit && is.null(LEARNERS[[learner_name]]$fit_nuisance_cf)) {
        message(sprintf("  [%s/%s/%s] SKIPPED — no fit_nuisance_cf", scenario, learner_name, cf_label))
        next
      }
      todo <- setdiff(OPTS$sims, completed_sims(output_file))
      message(sprintf("  [%s/%s/%s] %d sims to run", scenario, learner_name, cf_label, length(todo)))

      for (sim_id in todo) {
        sim_data <- read_sim(sim_id)
        X <- if (scenario == "simple") sim_data$C else sim_data$Z
        Y <- sim_data$Y; A <- sim_data$A; seed <- sim_data$seed

        nuisance <- tryCatch(
          estimate_nuisance(learner_name, Y, A, X, cross_fit, est_config, seed),
          error = function(e) {
            message(sprintf("    ERROR (nuisance, sim %d): %s", sim_id, conditionMessage(e)))
            list(pihat = rep(NA_real_, n), mu0hat = rep(NA_real_, n),
                 mu1hat = rep(NA_real_, n), pi_bounds = est_config$pi_bounds)
          })

        rows <- lapply(estimator_names, function(est_name) {
          r <- run_estimator(est_name, Y, A, X, nuisance)
          result_row(sim_id, seed, scenario, learner_name, cross_fit, est_name, r, nuisance)
        })
        append_rows(do.call(rbind, rows), output_file)
      }
    }
  }
}

# --- Oracle reference arm ----------------------------------------------------
for (scenario in OPTS$scenarios) {
  for (learner_name in oracle_learners) {
    files <- vapply(c(FALSE, TRUE), function(cf)
      file.path(out_dir, sprintf("%s_%s_%s_n%d.csv", scenario, learner_name,
                                 if (cf) "cf" else "nocf", n)), character(1))
    done <- Reduce(intersect, lapply(files, completed_sims))
    todo <- setdiff(OPTS$sims, done)
    message(sprintf("  [%s/%s] %d sims to run (both CF labels)", scenario, learner_name, length(todo)))

    for (sim_id in todo) {
      sim_data <- read_sim(sim_id)
      X <- if (scenario == "simple") sim_data$C else sim_data$Z
      Y <- sim_data$Y; A <- sim_data$A; seed <- sim_data$seed
      b <- est_config$pi_bounds

      pihat <- pmax(b[1], pmin(b[2], sim_data$pi_true))
      nuisance <- list(pihat = pihat, mu0hat = sim_data$mu0_true,
                       mu1hat = sim_data$mu1_true,
                       nuisance_time = 0, nuisance_time_wall = 0,
                       pi_bounds = b)

      rows <- lapply(estimator_names, function(est_name) {
        r <- run_estimator(est_name, Y, A, X, nuisance)
        result_row(sim_id, seed, scenario, learner_name, NA, est_name, r, nuisance)
      })
      df <- do.call(rbind, rows)
      for (i in seq_along(files)) {           # identical rows under both CF labels
        df$cross_fit <- c(FALSE, TRUE)[i]
        append_rows(df, files[i])
      }
    }
  }
}

message("R-learner track complete.")
}


# =============================================================================
# ==== Driver: TabPFN v3 track (estimators over saved Python nuisance fits)
# =============================================================================

# --- TabPFN v3 track: estimators over the Python nuisance fits ---------------
# python/tabpfn_v3_nuisance.py writes RAW nuisance vectors per (sim, scenario,
# cross_fit). This track truncates pihat to pi_bounds and pushes the vectors
# through the SAME four estimators as every R learner (same truncation policy,
# same tmle gbound), writing rows in the same per-config schema.
#
# Two variants are supported, distinguished by the nuisance directory the
# Python step wrote to:
#   tabpfn_v3_nuisance/      -> learner "tabpfn_v3"      (default, n_estimators = 8)
#   tabpfn_v3_ne1_nuisance/  -> learner "tabpfn_v3_ne1"  (no internal aggregation,
#                               n_estimators = 1; the manuscript's "TabPFN
#                               (no aggregation)" sensitivity arm at n = 1200)
#
# NAMING CORRESPONDENCE: because the TMLE here is R tmle::tmle() (logistic
# fluctuation — DEVIATIONS D5), the TMLE rows this track produces correspond
# to the production learner `tabpfn_v3_api_tmleR`, while the ipw/gcomp/aipw
# rows correspond to production `tabpfn_api_v3`.

if (OPTS$track %in% c("tabpfn", "all")) {

n        <- CONFIG$resolved$n
true_ate <- CONFIG$dgp$true_ate
data_dir <- CONFIG$resolved$data_dir
out_dir  <- file.path(CONFIG$resolved$results_dir, "per_config")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

bounds <- as.numeric(unlist(CONFIG$estimation$pi_bounds))
estimator_names <- unlist(CONFIG$estimators)

RESULT_COLS <- c("sim_id", "seed", "n", "scenario", "learner", "cross_fit",
                 "estimator", "estimate", "se", "ci_lower", "ci_upper",
                 "runtime", "nuisance_time", "nuisance_time_wall",
                 "true_ate", "bias", "covered")

variants <- list(
  list(dir = "tabpfn_v3_nuisance",     learner = "tabpfn_v3"),
  list(dir = "tabpfn_v3_ne1_nuisance", learner = "tabpfn_v3_ne1")
)
variants <- Filter(function(v)
  dir.exists(file.path(CONFIG$resolved$export_dir, v$dir)), variants)
if (length(variants) == 0)
  stop(sprintf("No TabPFN nuisance directories under %s — run python/tabpfn_v3_nuisance.py first.",
               CONFIG$resolved$export_dir))

for (v in variants) {
  nu_dir <- file.path(CONFIG$resolved$export_dir, v$dir)
  learner_name <- v$learner

  for (scenario in OPTS$scenarios) {
    for (cross_fit in OPTS$cross_fit) {
      cf_label <- if (cross_fit) "cf" else "nocf"
      output_file <- file.path(out_dir, sprintf("%s_%s_%s_n%d.csv",
                                                scenario, learner_name, cf_label, n))
      # Resume check mirrors the R track: a sim counts as done only if every
      # estimator row is present with a non-NA estimate; anything else is
      # dropped from the file and re-queued.
      existing <- if (file.exists(output_file)) {
        prev <- read.csv(output_file, stringsAsFactors = FALSE)
        ok <- vapply(split(prev, prev$sim_id),
                     function(d) all(estimator_names %in% d$estimator) && !any(is.na(d$estimate)),
                     logical(1))
        bad <- as.integer(names(ok)[!ok])
        if (length(bad) > 0) {
          message(sprintf("    re-queueing %d incomplete sims", length(bad)))
          write.csv(prev[!prev$sim_id %in% bad, , drop = FALSE], output_file, row.names = FALSE)
        }
        as.integer(names(ok)[ok])
      } else integer(0)

      n_done <- 0L
      for (sim_id in setdiff(OPTS$sims, existing)) {
        nu_file <- file.path(nu_dir, sprintf("sim_%04d_%s_%s.csv", sim_id, scenario, cf_label))
        if (!file.exists(nu_file)) next
        nu <- read.csv(nu_file)

        sim_data <- readRDS(file.path(data_dir, sprintf("sim_%04d.rds", sim_id)))
        stopifnot(nrow(nu) == sim_data$n, isTRUE(sim_data$n == n))
        X <- if (scenario == "simple") sim_data$C else sim_data$Z
        Y <- sim_data$Y; A <- sim_data$A; seed <- sim_data$seed

        nuisance <- list(pihat = pmax(bounds[1], pmin(bounds[2], nu$pihat_raw)),
                         mu0hat = nu$mu0hat, mu1hat = nu$mu1hat,
                         pi_bounds = bounds)

        rows <- lapply(estimator_names, function(est_name) {
          r <- run_estimator(est_name, Y, A, X, nuisance)
          bias <- r$estimate - true_ate
          data.frame(sim_id = sim_id, seed = seed, n = n, scenario = scenario,
                     learner = learner_name, cross_fit = cross_fit,
                     estimator = est_name, estimate = r$estimate, se = r$se,
                     ci_lower = r$ci_lower, ci_upper = r$ci_upper,
                     runtime = r$runtime, nuisance_time = NA_real_,
                     nuisance_time_wall = NA_real_,
                     true_ate = true_ate, bias = bias,
                     covered = if (!is.na(r$ci_lower) && !is.na(r$ci_upper))
                                 (r$ci_lower <= true_ate) & (true_ate <= r$ci_upper) else NA,
                     stringsAsFactors = FALSE)
        })
        df <- do.call(rbind, rows)[, RESULT_COLS]
        write.table(df, output_file, sep = ",", row.names = FALSE,
                    col.names = !file.exists(output_file), append = file.exists(output_file))
        n_done <- n_done + 1L
      }
      message(sprintf("  [%s/%s/%s] %d sims estimated", scenario, learner_name, cf_label, n_done))
    }
  }
}

message("TabPFN track complete.")
}
