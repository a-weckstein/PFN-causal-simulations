# =============================================================================
# 03_run_ate_estimators.R — IPW / G-computation / AIPW / TMLE on the saved
# nuisance fits (Simulation 1)
#
# For every (scenario x learner x cross-fit x sim) cell with a saved nuisance
# file (from 02_run_nuisance_learners.R, or python/tabpfn_v3_nuisance.py for
# tabpfn_v3): truncate pihat to pi_bounds and run the four estimators on the
# same (pihat, mu0hat, mu1hat) vectors.
#
# Writes one CSV per cell, one row per (sim, estimator):
#   results/n<k>/per_config/<scenario>_<learner>_<cf|nocf>_n<k>.csv
# Re-running replaces the rows for the requested sims and keeps the rest.
# =============================================================================

# --- Settings ----------------------------------------------------------------
# Command line:
#   Rscript 03_run_ate_estimators.R --n 1200
#   Rscript 03_run_ate_estimators.R --n 1200 --learners tabpfn_v3
#   flags:  --n 200|1200|5000   --learners a,b   --sims a:b|a,b,c
#           --scenarios simple,complex   --cross_fit true,false
# Interactive: setwd() to this directory, edit the values below, run top to bottom.

n         <- 200    # 200 / 1200 / 5000
learners  <- NULL   # NULL = roster in config.yaml (+ HAL variant for this n, + tabpfn_v3)
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
if (is.null(learners))  learners  <- c(CONFIG$learners, CONFIG$estimation$hal_by_n[[as.character(n)]],
                                       CONFIG$python_track$learner_name)
if (is.null(sims))      sims      <- seq_len(CONFIG$simulation$n_sims)
if (is.null(scenarios)) scenarios <- CONFIG$simulation$scenarios
if (is.null(cross_fit)) cross_fit <- CONFIG$estimation$cross_fit_options

pi_bounds       <- CONFIG$estimation$pi_bounds
true_ate        <- CONFIG$dgp$true_ate
estimator_names <- CONFIG$estimators
data_dir        <- file.path(PROJECT_ROOT, CONFIG$paths$data_inputs,    paste0("n", n))
nuisance_dir    <- file.path(PROJECT_ROOT, CONFIG$paths$data_processed, paste0("n", n), "nuisance")
results_dir     <- file.path(PROJECT_ROOT, CONFIG$paths$results,        paste0("n", n), "per_config")

suppressPackageStartupMessages({
  library(sandwich)
  library(tmle)
})


# =============================================================================
# ==== The four estimators (IPW / G-computation / AIPW / TMLE)
# =============================================================================
# All four consume the same nuisance list(pihat, mu0hat, mu1hat) from one
# learner, so estimator contrasts are on identical inputs.
#   ipw    stabilized Hajek weights in a weighted OLS of Y ~ A; HC sandwich SE
#   gcomp  plug-in mean(mu1hat - mu0hat); NO standard error by design
#   aipw   one-step efficient-influence-function estimator; SE = sd(psi)/sqrt(n)
#   tmle   tmle::tmle() with Q and g1W supplied; gbound is passed EXPLICITLY as
#          the study's pi_bounds — without it tmle() applies its own adaptive
#          floor 5/(sqrt(n)*ln n), which at n=200 (0.0667) is wider than the
#          study truncation [0.025, 0.975] and would re-truncate.
# Wald 95% CIs (estimate ± 1.96·SE) except tmle, which reports its own IC CI.

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
  # evalATT = FALSE skips tmle()'s unused ATT/ATC path (which can trigger a
  # costly internal SuperLearner refit of g); the ATE output is unchanged.
  result <- tmle(Y = Y, A = A, W = as.data.frame(X),
                 Q = cbind(nuisance$mu0hat, nuisance$mu1hat),
                 g1W = nuisance$pihat,
                 gbound = nuisance$pi_bounds,
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
  result$runtime <- el[["user.self"]] + el[["sys.self"]]
  result
}


# =============================================================================
# ==== Driver
# =============================================================================

read_sim <- function(sim_id) {
  f <- file.path(data_dir, sprintf("sim_%04d.rds", sim_id))
  if (!file.exists(f)) stop("Missing ", f, " - run 01_generate_dgp_data.R first")
  s <- readRDS(f)
  if (!isTRUE(s$n == n) || !isTRUE(s$true_ate == true_ate))
    stop("Cached ", f, " does not match config.yaml (n / true_ate)")
  s
}

# Script 02 saves one .rds per cell; the Python TabPFN track saves a .csv of
# raw (untruncated) vectors. Either way pihat is truncated to pi_bounds here.
read_nuisance <- function(learner_name, sim_id, scenario, cf_label) {
  stem <- file.path(nuisance_dir, learner_name,
                    sprintf("sim_%04d_%s_%s", sim_id, scenario, cf_label))
  if (file.exists(paste0(stem, ".rds"))) {
    nu <- readRDS(paste0(stem, ".rds"))
  } else if (file.exists(paste0(stem, ".csv"))) {
    d  <- read.csv(paste0(stem, ".csv"))
    nu <- list(pihat = d$pihat_raw, mu0hat = d$mu0hat, mu1hat = d$mu1hat,
               nuisance_time = NA_real_, nuisance_time_wall = NA_real_)
  } else {
    return(NULL)
  }
  nu$pihat     <- pmax(pi_bounds[1], pmin(pi_bounds[2], nu$pihat))
  nu$pi_bounds <- pi_bounds    # passed through to tmle() as gbound
  nu
}

result_row <- function(sim_id, seed, scenario, learner_name, cf, est_name, r, nu) {
  data.frame(
    sim_id = sim_id, seed = seed, n = n, scenario = scenario,
    learner = learner_name, cross_fit = cf, estimator = est_name,
    estimate = r$estimate, se = r$se, ci_lower = r$ci_lower, ci_upper = r$ci_upper,
    runtime = r$runtime,
    nuisance_time = nu$nuisance_time, nuisance_time_wall = nu$nuisance_time_wall,
    true_ate = true_ate, bias = r$estimate - true_ate,
    covered = if (!is.na(r$ci_lower) && !is.na(r$ci_upper))
                (r$ci_lower <= true_ate) & (true_ate <= r$ci_upper) else NA,
    stringsAsFactors = FALSE)
}

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

for (scenario in scenarios) {
  for (learner_name in learners) {
    for (cf in cross_fit) {
      cf_label <- if (cf) "cf" else "nocf"
      out_file <- file.path(results_dir, sprintf("%s_%s_%s_n%d.csv", scenario, learner_name, cf_label, n))

      rows <- list()
      for (sim_id in sims) {
        nu <- read_nuisance(learner_name, sim_id, scenario, cf_label)
        if (is.null(nu)) next
        sim_data <- read_sim(sim_id)
        stopifnot(length(nu$pihat) == n)
        X <- if (scenario == "simple") sim_data$C else sim_data$Z

        for (est_name in estimator_names) {
          r <- run_estimator(est_name, sim_data$Y, sim_data$A, X, nu)
          rows[[length(rows) + 1]] <- result_row(sim_id, sim_data$seed, scenario, learner_name,
                                                 cf, est_name, r, nu)
        }
      }
      message(sprintf("[%s/%s/%s] %d of %d sims estimated",
                      scenario, learner_name, cf_label, length(rows) / length(estimator_names), length(sims)))

      res <- do.call(rbind, rows)
      if (file.exists(out_file)) {          # keep rows for sims not requested in this run
        old <- read.csv(out_file, stringsAsFactors = FALSE)
        res <- rbind(old[!old$sim_id %in% sims, ], res)
      }
      if (is.null(res)) next
      write.csv(res[order(res$sim_id), ], out_file, row.names = FALSE)
    }
  }
}

message("Done.")
