# =============================================================================
# 02_run_nuisance_learners.R — fit the nuisance learners (Simulation 1)
#
# For every (scenario x learner x cross-fit x sim) cell: estimate the nuisance
# functions (propensity score + the two outcome regressions) and save the
# fitted vectors. Script 03 then runs the ATE estimators on them.
#
# Writes, per cell:
#   _data_processed/n<k>/nuisance/<learner>/sim_XXXX_<scenario>_<cf|nocf>.rds
#   = list(pihat, mu0hat, mu1hat, nuisance_time, nuisance_time_wall, ...)
# Resume-safe: cells whose file already exists are skipped.
#
# Learner roster (manuscript names in parentheses):
#   parametric (Parametric)   gam (GAM)            lasso (LASSO)
#   knn (KNN)                 bart (BART)          ranger (Random Forest)
#   xgboost_naimi (XGBoost)   hal_discrete (HAL, n=200/1200)
#   hal_n5000 (HAL, n=5000)   sl_default (SL Default)
#   sl_naimi_v2 (SL Naimi)    sl_balzer (SL Balzer)
# TabPFN (tabpfn_v3) is fit by python/tabpfn_v3_nuisance.py into the same
# directory layout.
# =============================================================================

# --- Settings ----------------------------------------------------------------
# Command line:
#   Rscript 02_run_nuisance_learners.R --n 1200
#   Rscript 02_run_nuisance_learners.R --n 1200 --learners parametric,ranger --sims 1:20
#   flags:  --n 200|1200|5000   --learners a,b   --sims a:b|a,b,c
#           --scenarios simple,complex   --cross_fit true,false
# Interactive: setwd() to this directory, edit the values below, run top to bottom.
# (In RStudio set Sys.setenv(HAL_CF_CORES = 1): the HAL learners fork via mclapply.)

n         <- 200    # 200 / 1200 / 5000
learners  <- NULL   # NULL = roster in config.yaml (+ the HAL variant for this n)
sims      <- NULL   # NULL = all sims; otherwise e.g. 1:20 or c(3, 7)
scenarios <- NULL   # NULL = both; otherwise "simple" and/or "complex"
cross_fit <- NULL   # NULL = both; otherwise TRUE and/or FALSE

# Command-line flags override the values above
args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0) {
  if (length(args) %% 2 != 0) stop("Usage: --flag value [--flag value ...]")
  opt <- setNames(args[c(FALSE, TRUE)], args[c(TRUE, FALSE)])
  unknown <- setdiff(names(opt), c("--n", "--learners", "--sims", "--scenarios", "--cross_fit"))
  if (length(unknown) > 0) stop("Unknown argument(s): ", paste(unknown, collapse = ", "))
  parse_sims <- function(s) {
    if (grepl(":", s)) { r <- as.integer(strsplit(s, ":")[[1]]); r[1]:r[2] }
    else as.integer(strsplit(s, ",")[[1]])
  }
  if (!is.na(opt["--n"]))         n         <- as.integer(opt["--n"])
  if (!is.na(opt["--learners"]))  learners  <- strsplit(opt["--learners"], ",")[[1]]
  if (!is.na(opt["--sims"]))      sims      <- parse_sims(opt["--sims"])
  if (!is.na(opt["--scenarios"])) scenarios <- strsplit(opt["--scenarios"], ",")[[1]]
  if (!is.na(opt["--cross_fit"])) cross_fit <- as.logical(strsplit(opt["--cross_fit"], ",")[[1]])
}

# --- Config and paths --------------------------------------------------------
PROJECT_ROOT <- {
  f <- grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(f) > 0) dirname(normalizePath(sub("--file=", "", f))) else getwd()
}
CONFIG <- yaml::read_yaml(file.path(PROJECT_ROOT, "config.yaml"))

if (!n %in% CONFIG$simulation$sample_sizes)
  stop("n must be one of: ", paste(CONFIG$simulation$sample_sizes, collapse = ", "))
if (is.null(learners))  learners  <- c(CONFIG$learners, CONFIG$estimation$hal_by_n[[as.character(n)]])
if (is.null(sims))      sims      <- seq_len(CONFIG$simulation$n_sims)
if (is.null(scenarios)) scenarios <- CONFIG$simulation$scenarios
if (is.null(cross_fit)) cross_fit <- CONFIG$estimation$cross_fit_options

n_folds      <- CONFIG$estimation$folds_by_n[[as.character(n)]]   # CF folds = learner-internal CV folds
data_dir     <- file.path(PROJECT_ROOT, CONFIG$paths$data_inputs,    paste0("n", n))
nuisance_dir <- file.path(PROJECT_ROOT, CONFIG$paths$data_processed, paste0("n", n), "nuisance")

suppressPackageStartupMessages({
  library(SuperLearner); library(gam); library(glmnet); library(caret)
  library(dbarts); library(ranger); library(xgboost); library(hal9001)
  library(tmle)       # provides the tmle.SL.dbarts* wrappers used by sl_default
  library(parallel)
})


# =============================================================================
# ==== RNG rules (sim_seed = starting_seed + sim_id; data seeded in script 01)
# =============================================================================
# CF folds    set.seed(sim_seed * 1000 + 7)   in generate_folds(); folds are shared by every learner, 
#                                              and this seed attempts to pin RNG stream for the CF fits that follow noCF fits, for further stability/reproducibility across runs
# there may still be some additional internal seeding/RNG streams for select learners

#' Balanced random fold assignment (1..n_splits)
generate_folds <- function(n, n_splits, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  sample(rep(1:n_splits, ceiling(n / n_splits))[1:n])
}

cf_fold_seed <- function(sim_seed) sim_seed * 1000 + 7
nocf_seed    <- function(sim_seed) sim_seed * 1000 + 11


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
# Cross-fitting convention: DML2. The CF path gives us
# out-of-fold nuisance vectors, then each estimator is solved once on the
# pooled vectors (no per-fold ATE averaging). Fold assignments come from
# generate_folds() with a learner-independent seed, so all learners should share the
# same folds within a simulation and remain per-sim paired.

LEARNERS <- list()

register_learner <- function(name, fit_nuisance, fit_nuisance_cf) {
  LEARNERS[[name]] <<- list(name = name,
                            fit_nuisance = fit_nuisance,
                            fit_nuisance_cf = fit_nuisance_cf)
  invisible(NULL)
}

#' Estimate nuisance functions for one (learner, dataset, cross_fit) triplet.
estimate_nuisance <- function(learner_name, Y, A, X, cross_fit, config, sim_seed) {
  learner <- LEARNERS[[learner_name]]
  if (is.null(learner)) stop(sprintf("Learner '%s' not loaded", learner_name))
  # hal_n5000's cross-fit path derives its per-fold seeds from the sim seed.
  config$sim_seed <- sim_seed

  pt <- proc.time()
  if (cross_fit) {
    fold_ids <- generate_folds(length(Y), config$n_splits,
                               seed = cf_fold_seed(sim_seed))
    result <- learner$fit_nuisance_cf(Y, A, X, fold_ids, config)
  } else {
    set.seed(nocf_seed(sim_seed))
    result <- learner$fit_nuisance(Y, A, X, config)
  }
  el <- proc.time() - pt

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
# ==== Shared SuperLearner helpers (sl_default, sl_naimi_v2, sl_balzer)
# =============================================================================
# Convex NNLS SuperLearner (V = config$cv_folds); fit order propensity -> mu0
# -> mu1 fixes the RNG consumption order.

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

# --- Naimi shared candidate library ------------------------------------------
# 10 candidates via create.Learner (registered once):
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
# ==== Learner: parametric — logistic PS + single linear Y ~ A + X outcome
# ==== (the one non-arm-stratified outcome fjit in the study)
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
# ==== Learner: gam — SL.gam called directly (no ensemble), deg.gam = 4
# =============================================================================

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

register_learner("gam", fit_nuisance_gam, fit_nuisance_cf_gam)


# =============================================================================
# ==== Learner: lasso — cv.glmnet (alpha = 1, lambda.min), main effects + 2-way interactions
# =============================================================================

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

register_learner("lasso", fit_nuisance_lasso, fit_nuisance_cf_lasso)


# =============================================================================
# ==== Learner: knn — caret::knnreg, standardized, k CV-selected from {5,...,50}
# =============================================================================

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

register_learner("knn", fit_nuisance_knn, fit_nuisance_cf_knn)


# =============================================================================
# ==== Learner: bart — dbarts::bart (probit PS / Gaussian outcomes), content-seeded
# =============================================================================

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

register_learner("bart", fit_nuisance_bart, fit_nuisance_cf_bart)


# =============================================================================
# ==== Learner: ranger — 500 trees, mtry = 2, min.node.size CV-selected from {30, 60}
# =============================================================================

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

register_learner("ranger", fit_nuisance_ranger, fit_nuisance_cf_ranger)


# =============================================================================
# ==== Learner: xgboost_naimi — 500 rounds, depth 4, eta 0.1, min_child_weight CV {30, 60}
# =============================================================================

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
                 fit_nuisance_cf_xgboost_naimi)


# =============================================================================
# ==== Learner: hal_discrete — discrete SL over fit_hal(d2, s0) vs (d2, s1) by CV risk
# =============================================================================
# CF folds run in parallel (mclapply, deterministic L'Ecuyer-CMRG streams;
# HAL_CF_CORES caps forks).

.fit_hal_discrete <- function(X_mat, Y_vec, family) {
  configs <- list(
    list(smoothness_orders = 0, label = "HAL(s=0,d=2)"),
    list(smoothness_orders = 1, label = "HAL(s=1,d=2)")
  )

  best_risk <- Inf
  best_fit <- NULL

  for (cfg in configs) {
    fit <- tryCatch(
      fit_hal(X = X_mat, Y = Y_vec, family = family, max_degree = 2,
              smoothness_orders = cfg$smoothness_orders),
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

  ps_result <- .fit_hal_discrete(X_mat, as.numeric(A), "binomial")
  pihat <- as.numeric(predict(ps_result$fit, new_data = X_mat, type = "response"))
  pihat <- pmax(bounds[1], pmin(bounds[2], pihat))

  idx_0 <- A == 0
  Q0_result <- .fit_hal_discrete(X_mat[idx_0, , drop = FALSE], Y[idx_0], "gaussian")
  mu0hat <- as.numeric(predict(Q0_result$fit, new_data = X_mat))

  idx_1 <- A == 1
  Q1_result <- .fit_hal_discrete(X_mat[idx_1, , drop = FALSE], Y[idx_1], "gaussian")
  mu1hat <- as.numeric(predict(Q1_result$fit, new_data = X_mat))

  list(pihat = pihat, mu0hat = mu0hat, mu1hat = mu1hat)
}

.fit_one_fold_hal_discrete <- function(k, Y, A, X_mat, fold_ids, bounds) {
  train <- fold_ids != k
  test  <- fold_ids == k
  X_tr  <- X_mat[train, , drop = FALSE]
  X_te  <- X_mat[test,  , drop = FALSE]
  Y_tr  <- Y[train]
  A_tr  <- A[train]

  ps_res <- .fit_hal_discrete(X_tr, as.numeric(A_tr), "binomial")
  pihat_k <- as.numeric(predict(ps_res$fit, new_data = X_te, type = "response"))
  pihat_k <- pmax(bounds[1], pmin(bounds[2], pihat_k))

  i0 <- which(A_tr == 0)
  Q0_res <- .fit_hal_discrete(X_tr[i0, , drop = FALSE], Y_tr[i0], "gaussian")
  mu0_k  <- as.numeric(predict(Q0_res$fit, new_data = X_te))

  i1 <- which(A_tr == 1)
  Q1_res <- .fit_hal_discrete(X_tr[i1, , drop = FALSE], Y_tr[i1], "gaussian")
  mu1_k  <- as.numeric(predict(Q1_res$fit, new_data = X_te))

  list(test = which(test), pihat_k = pihat_k, mu0_k = mu0_k, mu1_k = mu1_k)
}

fit_nuisance_cf_hal_discrete <- function(Y, A, X, fold_ids, config) {
  X_mat  <- as.matrix(X)
  bounds <- config$pi_bounds
  n      <- length(Y)
  K      <- max(fold_ids)

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
    Y = Y, A = A, X_mat = X_mat, fold_ids = fold_ids, bounds = bounds,
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
                 fit_nuisance_cf_hal_discrete)


# =============================================================================
# ==== Learner: sl_default — the tmle package's default SL libraries
# =============================================================================
#   propensity: SL.glm + tmle.SL.dbarts.k.5 + SL.gam
#   outcome:    SL.glm + tmle.SL.dbarts2    + SL.glmnet

.SL_DEFAULT_PS  <- c("SL.glm", "tmle.SL.dbarts.k.5", "SL.gam")
.SL_DEFAULT_OUT <- c("SL.glm", "tmle.SL.dbarts2", "SL.glmnet")

register_learner(
  "sl_default",
  function(Y, A, X, config)
    sl_fit_nuisance(Y, A, X, config, .SL_DEFAULT_PS, .SL_DEFAULT_OUT),
  function(Y, A, X, fold_ids, config)
    sl_fit_nuisance_cf(Y, A, X, fold_ids, config, .SL_DEFAULT_PS, .SL_DEFAULT_OUT))


# =============================================================================
# ==== Learner: sl_naimi_v2 — the Naimi library on covariates augmented with
# ==== all pairwise interaction features
# =============================================================================

register_learner(
  "sl_naimi_v2",
  function(Y, A, X, config)
    sl_fit_nuisance(Y, A, X, config, create_naimi_library(),
                    transform_X = add_pairwise_interactions),
  function(Y, A, X, fold_ids, config)
    sl_fit_nuisance_cf(Y, A, X, fold_ids, config, create_naimi_library(),
                       transform_X = add_pairwise_interactions))


# =============================================================================
# ==== Learner: sl_balzer — Balzer-style SuperLearner library
# =============================================================================

.SL_BALZER <- c("SL.glm", "SL.step.interaction", "SL.earth", "SL.mean")

register_learner(
  "sl_balzer",
  function(Y, A, X, config)
    sl_fit_nuisance(Y, A, X, config, .SL_BALZER),
  function(Y, A, X, fold_ids, config)
    sl_fit_nuisance_cf(Y, A, X, fold_ids, config, .SL_BALZER))


# =============================================================================
# ==== Learner: hal_n5000 — single fit_hal(d2, s1), the study's HAL at n = 5000 (runtime)
# =============================================================================
# CF folds run in parallel (mclapply; HAL_CF_CORES caps forks) with
# deterministic per-fold seeds (sim_seed*1000 + 70 + k).

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

register_learner("hal_n5000", fit_nuisance_hal_n5000, fit_nuisance_cf_hal_n5000)


# =============================================================================
# ==== Driver
# =============================================================================

unknown <- setdiff(learners, names(LEARNERS))
if (length(unknown) > 0) stop("Unknown learner(s): ", paste(unknown, collapse = ", "))

fit_config <- list(pi_bounds = CONFIG$estimation$pi_bounds,
                   n_splits  = n_folds,
                   cv_folds  = n_folds)

read_sim <- function(sim_id) {
  f <- file.path(data_dir, sprintf("sim_%04d.rds", sim_id))
  if (!file.exists(f)) stop("Missing ", f, " - run 01_generate_dgp_data.R first")
  s <- readRDS(f)
  if (!isTRUE(s$n == n) || !isTRUE(s$true_ate == CONFIG$dgp$true_ate))
    stop("Cached ", f, " does not match config.yaml (n / true_ate)")
  s
}

for (scenario in scenarios) {
  for (learner_name in learners) {
    for (cf in cross_fit) {
      cf_label  <- if (cf) "cf" else "nocf"
      out_dir   <- file.path(nuisance_dir, learner_name)
      out_files <- file.path(out_dir, sprintf("sim_%04d_%s_%s.rds", sims, scenario, cf_label))
      dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

      todo <- which(!file.exists(out_files))
      message(sprintf("[%s/%s/%s] %d of %d sims to fit",
                      scenario, learner_name, cf_label, length(todo), length(sims)))

      for (i in todo) {
        sim_data <- read_sim(sims[i])
        X <- if (scenario == "simple") sim_data$C else sim_data$Z
        nuisance <- tryCatch(
          estimate_nuisance(learner_name, sim_data$Y, sim_data$A, X, cf, fit_config, sim_data$seed),
          error = function(e) {
            message(sprintf("  sim %d FAILED: %s", sims[i], conditionMessage(e)))
            NULL
          })
        if (is.null(nuisance)) next
        saveRDS(nuisance, paste0(out_files[i], ".tmp"))
        file.rename(paste0(out_files[i], ".tmp"), out_files[i])   # no half-written files if interrupted
      }
    }
  }
}

message("Done.")
