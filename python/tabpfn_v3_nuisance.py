"""
tabpfn_v3_nuisance.py — TabPFN v3 cloud-API nuisance fits (learner "tabpfn_v3")

Reads the R-exported datasets (_data_processed/n<k>/sim_XXXX.csv, identical
realizations to the R track's RDS cache, including the shared cross-fitting
fold assignment in `fold_id`) and writes raw (untruncated) nuisance vectors
per (sim, scenario, cross_fit). Estimation then happens in R:
    Rscript 03_run_ate_estimators.R --n <k> --learners tabpfn_v3

Model: tabpfn-client == 0.3.0 (pinned in requirements.txt), ModelVersion.V3,
n_estimators = 8, random_state = 0 (deterministic). Classifier for the
propensity; one regressor per treatment arm for the outcomes.

Usage:
    python tabpfn_v3_nuisance.py --n 200 [--sims 1-200] [--scenarios simple,complex]
Requires a one-time `tabpfn_client.init()` browser login in this environment.
Resume-safe: existing output files are skipped.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

N_ESTIMATORS = 8
RANDOM_STATE = 0

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, required=True, help="sample size (200/1200/5000)")
    p.add_argument("--sims", type=str, default="1-200", help="e.g. 1-200 or 1,5,7")
    p.add_argument("--scenarios", type=str, default="simple,complex")
    p.add_argument("--cross_fit", type=str, default="nocf,cf")
    return p.parse_args()


def sim_id_list(spec):
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return sorted({int(s) for s in spec.split(",")})


def make_models():
    from tabpfn_client import TabPFNClassifier, TabPFNRegressor
    from tabpfn_client.estimator import ModelVersion

    def clf():
        return TabPFNClassifier.create_default_for_version(
            ModelVersion.V3, n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)

    def reg():
        return TabPFNRegressor.create_default_for_version(
            ModelVersion.V3, n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)

    return clf, reg


def fit_nuisance(Y, A, X, make_clf, make_reg):
    """Full-sample fits; returns raw pihat (no truncation) + arm-stratified mus."""
    clf = make_clf()
    clf.fit(X, A)
    pihat = clf.predict_proba(X)[:, 1]

    mu = {}
    for arm in (0, 1):
        reg = make_reg()
        idx = A == arm
        reg.fit(X[idx], Y[idx])
        mu[arm] = np.asarray(reg.predict(X), dtype=np.float64)

    return pihat, mu[0], mu[1]


def fit_nuisance_cf(Y, A, X, fold_ids, make_clf, make_reg):
    """Out-of-fold fits using the R-exported shared fold assignment."""
    n = len(Y)
    pihat = np.full(n, np.nan)
    mu0 = np.full(n, np.nan)
    mu1 = np.full(n, np.nan)

    for k in sorted(np.unique(fold_ids)):
        train, test = fold_ids != k, fold_ids == k
        clf = make_clf()
        clf.fit(X[train], A[train])
        pihat[test] = clf.predict_proba(X[test])[:, 1]

        for arm, out in ((0, mu0), (1, mu1)):
            idx = train & (A == arm)
            reg = make_reg()
            reg.fit(X[idx], Y[idx])
            out[test] = np.asarray(reg.predict(X[test]), dtype=np.float64)

    return pihat, mu0, mu1


def main():
    args = parse_args()
    data_dir = PROJECT_ROOT / "_data_processed" / f"n{args.n}"
    if not data_dir.exists():
        sys.exit(f"Missing {data_dir} — run `Rscript 01_generate_dgp_data.R --n {args.n}` first.")
    out_dir = data_dir / "nuisance" / "tabpfn_v3"   # same layout as script 02's R learners
    out_dir.mkdir(parents=True, exist_ok=True)

    make_clf, make_reg = make_models()
    scenarios = args.scenarios.split(",")
    cf_opts = args.cross_fit.split(",")
    cols = {"simple": ["C1", "C2", "C3", "C4"], "complex": ["Z1", "Z2", "Z3", "Z4"]}

    for sim_id in sim_id_list(args.sims):
        df = pd.read_csv(data_dir / f"sim_{sim_id:04d}.csv")
        Y = df["Y"].to_numpy(dtype=np.float64)
        A = df["A"].to_numpy(dtype=int)
        fold_ids = df["fold_id"].to_numpy(dtype=int)

        for scenario in scenarios:
            X = df[cols[scenario]].to_numpy(dtype=np.float64)
            for cf in cf_opts:
                out_file = out_dir / f"sim_{sim_id:04d}_{scenario}_{cf}.csv"
                if out_file.exists():
                    continue
                t0 = time.time()
                if cf == "cf":
                    pihat, mu0, mu1 = fit_nuisance_cf(Y, A, X, fold_ids, make_clf, make_reg)
                else:
                    pihat, mu0, mu1 = fit_nuisance(Y, A, X, make_clf, make_reg)
                elapsed = time.time() - t0
                tmp = out_file.with_suffix(".tmp")
                pd.DataFrame({"pihat_raw": pihat, "mu0hat": mu0, "mu1hat": mu1}
                             ).to_csv(tmp, index=False)
                tmp.rename(out_file)   # atomic: no half-written outputs on interrupt
                print(f"sim {sim_id:4d} {scenario:7s} {cf:4s} ({elapsed:.1f}s)", flush=True)

    print("Done.")


if __name__ == "__main__":
    main()
