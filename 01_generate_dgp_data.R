# =============================================================================
# 01_generate_dgp_data.R — generate + cache the simulated datasets (Simulation 1)
#
# DGP from Naimi, Mishler & Kennedy (2023), originally adapted from Kang & Schafer (2007)
# Writes (per sim):
#   _data_inputs/n<k>/sim_XXXX.rds     full dataset incl. ground truth (R track)
#   _data_processed/n<k>/sim_XXXX.csv  covariates/treatment/outcome + the shared
#                                      CF fold assignment (Python TabPFN track)
# Idempotent: existing sims are validated against config.yaml, not rewritten.
# =============================================================================

# --- Settings ----------------------------------------------------------------
# Command line:
#   Rscript 01_generate_dgp_data.R --n 1200
#   Rscript 01_generate_dgp_data.R --n 200 --sims 1:5
# Interactive: setwd() to this directory, edit the values below, run top to bottom.

n    <- 200    # 200 / 1200 / 5000
sims <- NULL   # NULL = all sims; otherwise e.g. 1:20 or c(3, 7)

# Command-line flags override the values above
args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0) {
  if (length(args) %% 2 != 0) stop("Usage: --flag value [--flag value ...]")
  opt <- setNames(args[c(FALSE, TRUE)], args[c(TRUE, FALSE)])
  unknown <- setdiff(names(opt), c("--n", "--sims"))
  if (length(unknown) > 0) stop("Unknown argument(s): ", paste(unknown, collapse = ", "))
  parse_sims <- function(s) {
    if (grepl(":", s)) { r <- as.integer(strsplit(s, ":")[[1]]); r[1]:r[2] }
    else as.integer(strsplit(s, ",")[[1]])
  }
  if (!is.na(opt["--n"]))    n    <- as.integer(opt["--n"])
  if (!is.na(opt["--sims"])) sims <- parse_sims(opt["--sims"])
}

# --- Config and paths --------------------------------------------------------
PROJECT_ROOT <- {
  f <- grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(f) > 0) dirname(normalizePath(sub("--file=", "", f))) else getwd()
}
CONFIG <- yaml::read_yaml(file.path(PROJECT_ROOT, "config.yaml"))

if (!n %in% CONFIG$simulation$sample_sizes)
  stop("n must be one of: ", paste(CONFIG$simulation$sample_sizes, collapse = ", "))
if (is.null(sims)) sims <- seq_len(CONFIG$simulation$n_sims)

n_folds    <- CONFIG$estimation$folds_by_n[[as.character(n)]]
data_dir   <- file.path(PROJECT_ROOT, CONFIG$paths$data_inputs,    paste0("n", n))
export_dir <- file.path(PROJECT_ROOT, CONFIG$paths$data_processed, paste0("n", n))


# =============================================================================
# ==== RNG rules (sim_seed = starting_seed + sim_id)
# =============================================================================
# data        set.seed(sim_seed)              in simulate_data()
# CF folds    set.seed(sim_seed * 1000 + 7)   in generate_folds(); folds are
#                                             shared by every learner
# exported fold_id column gives Py TabPFN track the same folds

#' Balanced random fold assignment (1..n_splits), deterministic given seed
generate_folds <- function(n, n_splits, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  sample(rep(1:n_splits, ceiling(n / n_splits))[1:n])
}

cf_fold_seed <- function(sim_seed) sim_seed * 1000 + 7


# =============================================================================
# ==== Data-generating process (Naimi, Mishler & Kennedy 2023 / Kang & Schafer 2007)
# =============================================================================
# 4 iid standard-normal confounders C1-C4, binary treatment (logistic in C),
# continuous outcome (linear in C, homogeneous effect), true ATE = 6.
# The "complex" scenario gives analyst transformed covariates, Z:
# Z = h(C) instead of C:
#   Z1 = exp(C1/2)                    Z2 = C2 / (1 + exp(C1)) + 10
#   Z3 = (C1*C3/25 + 0.6)^3           Z4 = (C2 + C4 + 20)^2
# simulate_data(n, seed) is deterministic given seed. 
# note this corresponds to Naimi et al's manuscript description of the DGP, not (necessarily) to their accompanying git code

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

#' Generate one complete simulated dataset
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

dir.create(data_dir,   recursive = TRUE, showWarnings = FALSE)
dir.create(export_dir, recursive = TRUE, showWarnings = FALSE)

n_new <- 0L; n_existing <- 0L
for (sim_id in sims) {
  seed     <- CONFIG$simulation$starting_seed + sim_id
  rds_file <- file.path(data_dir,   sprintf("sim_%04d.rds", sim_id))
  csv_file <- file.path(export_dir, sprintf("sim_%04d.csv", sim_id))

  if (file.exists(rds_file)) {
    sim_data <- readRDS(rds_file)
    if (!isTRUE(sim_data$n == n) || !isTRUE(sim_data$true_ate == CONFIG$dgp$true_ate) ||
        !isTRUE(sim_data$seed == seed))
      stop(sprintf("Cached %s does not match config (n=%s, true_ate=%s, seed=%s); remove stale cache.",
                   basename(rds_file), sim_data$n, sim_data$true_ate, sim_data$seed))
    n_existing <- n_existing + 1L
  } else {
    sim_data <- simulate_data(n = n, seed = seed, dgp_params = CONFIG$dgp)
    saveRDS(sim_data, rds_file)
    n_new <- n_new + 1L
  }

  if (!file.exists(csv_file)) {
    fold_ids <- generate_folds(n, n_folds, seed = cf_fold_seed(seed))
    write.csv(data.frame(sim_data$C, sim_data$Z,
                         A = as.integer(sim_data$A), Y = sim_data$Y,
                         fold_id = as.integer(fold_ids)),
              csv_file, row.names = FALSE)
  }
}

message(sprintf("%d generated, %d cached -> %s", n_new, n_existing, data_dir))
