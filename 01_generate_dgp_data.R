# =============================================================================
# 01_generate_dgp_data.R — generate + cache the simulated datasets
#
# Simulation (1) of the manuscript: the Kang & Schafer (2007) data-generating
# process as adapted by Naimi, Mishler & Kennedy (2023). Writes, per sim:
#   _data_inputs/n<k>/sim_XXXX.rds     full dataset incl. ground truth (R track)
#   _data_processed/n<k>/sim_XXXX.csv  covariates/treatment/outcome + the shared
#                                      CF fold assignment (Python TabPFN track)
#
# The CSV is an exact export of the RDS (same realizations), so the Python
# track is evaluated on byte-identical data and its per-sim rows pair with the
# R learners'. The fold_id column reproduces script 02's generate_folds()
# call (same seed rule), so the Python CF track uses the SAME folds as every
# R learner. Both files are idempotent: existing sims are validated, not
# rewritten.
#
#   Rscript 01_generate_dgp_data.R --n 1200 [--sims 1:200]
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
# ==== Data-generating process
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
# ==== Driver: simulate, cache, export
# =============================================================================



n          <- CONFIG$resolved$n
n_splits   <- CONFIG$resolved$n_splits
data_dir   <- CONFIG$resolved$data_dir
export_dir <- CONFIG$resolved$export_dir
dir.create(data_dir,   recursive = TRUE, showWarnings = FALSE)
dir.create(export_dir, recursive = TRUE, showWarnings = FALSE)

n_new <- 0L; n_existing <- 0L
for (sim_id in OPTS$sims) {
  seed     <- CONFIG$simulation$starting_seed + sim_id
  rds_file <- file.path(data_dir,   sprintf("sim_%04d.rds", sim_id))
  csv_file <- file.path(export_dir, sprintf("sim_%04d.csv", sim_id))

  if (file.exists(rds_file)) {
    cached <- readRDS(rds_file)
    if (!isTRUE(cached$n == n) || !isTRUE(cached$true_ate == CONFIG$dgp$true_ate) ||
        !isTRUE(cached$seed == seed))
      stop(sprintf("Cached %s does not match config (n=%s, true_ate=%s, seed=%s); remove stale cache.",
                   basename(rds_file), cached$n, cached$true_ate, cached$seed))
    sim_data <- cached
    n_existing <- n_existing + 1L
  } else {
    sim_data <- simulate_data(n = n, seed = seed, dgp_params = CONFIG$dgp)
    saveRDS(sim_data, rds_file)
    n_new <- n_new + 1L
  }

  if (!file.exists(csv_file)) {
    fold_ids <- generate_folds(n, n_splits, seed = cf_fold_seed(seed))
    write.csv(data.frame(sim_data$C, sim_data$Z,
                         A = as.integer(sim_data$A), Y = sim_data$Y,
                         fold_id = as.integer(fold_ids)),
              csv_file, row.names = FALSE)
  }
}

message(sprintf("Stage 1: %d generated, %d already cached -> %s", n_new, n_existing, data_dir))

# Validation: first requested sim
check <- readRDS(file.path(data_dir, sprintf("sim_%04d.rds", OPTS$sims[1])))
message(sprintf("  sim %d: n=%d, treated=%.3f, mean(pi)=%.3f, mean(Y)=%.1f, sample ATE=%.3f",
                OPTS$sims[1], check$n, mean(check$A), mean(check$pi_true),
                mean(check$Y), mean(check$Y1 - check$Y0)))
