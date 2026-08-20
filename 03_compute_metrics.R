# =============================================================================
# 03_compute_metrics.R — combine per-sim results and compute summary metrics
#
# Outputs (results/n<k>/):
#   all_sim_results_n<k>.csv   one row per (sim, scenario, learner, cross_fit,
#                              estimator) — the canonical per-sim file
#   summary_metrics_n<k>.csv   one row per cell with n_sims, n_failed,
#                              mean_bias (+ Monte-Carlo SE), empirical SE,
#                              mean model SE, SE ratio, RMSE, coverage
#                              (+ binomial MC-SE), mean nuisance time
#
#   Rscript 03_compute_metrics.R --n 1200
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
# ==== Combine + summarize
# =============================================================================


results_dir <- CONFIG$resolved$results_dir
per_config  <- file.path(results_dir, "per_config")
true_ate    <- CONFIG$dgp$true_ate
n           <- CONFIG$resolved$n

files <- list.files(per_config, pattern = "\\.csv$", full.names = TRUE)
if (length(files) == 0) stop("No per-config results found — run script 02 first.")

all_results <- do.call(rbind, lapply(files, read.csv, stringsAsFactors = FALSE))

# Guard against duplicate rows (e.g. interrupted append + resume)
key <- with(all_results, paste(sim_id, scenario, learner, cross_fit, estimator))
if (any(duplicated(key))) {
  warning(sprintf("Dropping %d duplicated result rows", sum(duplicated(key))))
  all_results <- all_results[!duplicated(key), ]
}

per_sim_file <- file.path(results_dir, sprintf("all_sim_results_n%d.csv", n))
write.csv(all_results, per_sim_file, row.names = FALSE)

cells <- split(all_results,
               with(all_results, interaction(scenario, learner, cross_fit,
                                             estimator, drop = TRUE)))
summary_df <- do.call(rbind, lapply(cells, function(d) {
  n_sims   <- nrow(d)
  n_failed <- sum(is.na(d$estimate))
  emp_se   <- sd(d$estimate, na.rm = TRUE)
  cover    <- mean(d$covered, na.rm = TRUE)
  n_cov    <- sum(!is.na(d$covered))
  data.frame(
    n = n,
    scenario = d$scenario[1], learner = d$learner[1],
    cross_fit = d$cross_fit[1], estimator = d$estimator[1],
    n_sims = n_sims, n_failed = n_failed,
    mean_estimate = mean(d$estimate, na.rm = TRUE),
    mean_bias = mean(d$bias, na.rm = TRUE),
    mcse_bias = emp_se / sqrt(n_sims - n_failed),
    empirical_se = emp_se,
    mean_se = mean(d$se, na.rm = TRUE),
    se_ratio = if (!is.na(emp_se) && emp_se > 0)
                 mean(d$se, na.rm = TRUE) / emp_se else NA_real_,
    rmse = sqrt(mean(d$bias^2, na.rm = TRUE)),
    coverage = if (n_cov > 0) cover else NA_real_,
    mcse_coverage = if (n_cov > 0) sqrt(cover * (1 - cover) / n_cov) else NA_real_,
    mean_nuisance_time = mean(d$nuisance_time, na.rm = TRUE),
    stringsAsFactors = FALSE)
}))
summary_df <- summary_df[order(summary_df$scenario, summary_df$learner,
                               summary_df$cross_fit, summary_df$estimator), ]
rownames(summary_df) <- NULL

summary_file <- file.path(results_dir, sprintf("summary_metrics_n%d.csv", n))
write.csv(summary_df, summary_file, row.names = FALSE)

message(sprintf("Stage 3: %d per-sim rows -> %s", nrow(all_results), per_sim_file))
message(sprintf("         %d summary cells -> %s", nrow(summary_df), summary_file))

# Validation printout: cells short of the full sim count
expected <- CONFIG$simulation$n_sims
short <- summary_df[summary_df$n_sims < expected, ]
if (nrow(short) > 0) {
  message(sprintf("  NOTE: %d cells have < %d sims (in-progress or failed):",
                  nrow(short), expected))
  for (i in seq_len(min(nrow(short), 20)))
    message(sprintf("    %s / %s / cf=%s / %s: %d sims",
                    short$scenario[i], short$learner[i], short$cross_fit[i],
                    short$estimator[i], short$n_sims[i]))
}
