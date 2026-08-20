"""
tabpfn_v3_nuisance.py — TabPFN v3 cloud-API nuisance fits (learner "tabpfn_v3")

This is the ONLY Python step in the study. It reads the R-exported datasets
(_data_processed/n<k>/sim_XXXX.csv — identical realizations to the R track's
RDS cache, including the shared cross-fitting fold assignment in `fold_id`)
and writes RAW (untruncated) nuisance vectors per (sim, scenario, cross_fit):

    _data_processed/n<k>/tabpfn_v3_nuisance/sim_XXXX_<scenario>_<nocf|cf>.csv
        columns: pihat_raw, mu0hat, mu1hat  (row-aligned with the dataset)

Estimation then happens in R (Rscript run.R --n <k> --stage 2b), through the
SAME four estimators as every R learner — including propensity truncation to
pi_bounds and tmle::tmle() with explicit gbound — so the TabPFN arm differs
from the R learners in nothing but the nuisance model.

Model: tabpfn-client == 0.3.0 (pinned in requirements.txt — the study spec),
ModelVersion.V3, n_estimators = 8, random_state = 0 (deterministic).
Classifier for the propensity, regressor per treatment arm for the outcomes
(arm-stratified, like the R learners).

Transient-failure retry (ported 2026-07-15 from the ACIC production module
`ACIC_2016/analysis/modules/python/tabpfn_nuisance_v3.py`; DEVIATIONS D10)
---------------------------------------------------------------------------
The v3 cloud API is intermittently flaky: individual .fit()/.predict() calls
raise RuntimeError("TabPFN is inaccessible at the moment, please try again
later.") during short server down-windows (seconds to low minutes; a 31-min
outage has been observed). Without retry, one blip discards a whole
(sim, scenario, cf) unit of work. Every cloud call therefore goes through
`_retry()`, which retries ONLY recognized-transient failures with capped
deterministic exponential backoff. Genuine errors (auth, credit/quota,
data/shape) re-raise immediately, and a transient error that outlives the
budget still re-raises, so the per-file resume (skip-existing) remains the
backstop. Retrying is reproducibility-safe: `random_state` is fixed, so a
retried fit is bit-identical — only WHETHER a call succeeds changes, never
its result. Tunable via env: TABPFN_RETRY_TRIES (default 6),
TABPFN_RETRY_BASE seconds (default 5), TABPFN_RETRY_CAP seconds (default 120).

Usage:
    python tabpfn_v3_nuisance.py --n 200 [--sims 1-200] [--scenarios simple,complex]
Requires a one-time `tabpfn_client.init()` browser login in this environment.
Resume-safe: existing output files are skipped.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

N_ESTIMATORS = 8
RANDOM_STATE = 0

# --- Transient-failure retry around cloud API calls --------------------------
_RETRY_TRIES = int(os.environ.get("TABPFN_RETRY_TRIES", "6"))
_RETRY_BASE = float(os.environ.get("TABPFN_RETRY_BASE", "5.0"))    # seconds
_RETRY_CAP = float(os.environ.get("TABPFN_RETRY_CAP", "120.0"))    # max single sleep

# Substrings (lowercased match) marking a retryable transient server/network
# condition. The canonical one is "inaccessible ... please try again later".
_TRANSIENT_MARKERS = (
    "inaccessible", "try again", "temporarily", "unavailable",
    "timeout", "timed out", "connection", "reset by peer",
    "remote end closed", "max retries", "bad gateway", "gateway",
    "rate limit", "too many requests", "429",
    "500", "502", "503", "504", "server error",
)
# Substrings marking a FATAL condition — never retry (retrying only wastes time
# and, for credit exhaustion, burns the retry budget without any chance of
# success). These take precedence over the transient markers.
_FATAL_MARKERS = (
    "unauthorized", "authentication", "invalid token", "forbidden",
    "401", "403", "quota", "credit", "insufficient", "payment", "402",
    "usage limit",
)


def _is_transient(exc):
    """True iff `exc` looks like a retryable transient API/network failure."""
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    msg = str(exc).lower()
    if any(m in msg for m in _FATAL_MARKERS):
        return False
    return any(m in msg for m in _TRANSIENT_MARKERS)


def _retry(fn, *args, _what="call", **kwargs):
    """Call fn(*args, **kwargs), retrying recognized-transient failures with
    capped, deterministic exponential backoff (no RNG, so nothing here perturbs
    the seeded TabPFN fits). Fatal or unrecognized errors re-raise immediately;
    a transient error re-raises after TABPFN_RETRY_TRIES attempts."""
    last = None
    for i in range(_RETRY_TRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - classified then re-raised
            last = e
            if not _is_transient(e) or i == _RETRY_TRIES - 1:
                raise
            sleep_s = (min(_RETRY_CAP, _RETRY_BASE * (2 ** i))
                       + 0.25 * _RETRY_BASE * (i % 4))
            print(
                f"    [tabpfn-retry] {_what}: transient failure "
                f"({type(e).__name__}: {str(e)[:80]}); "
                f"attempt {i + 1}/{_RETRY_TRIES}, sleeping {sleep_s:.0f}s",
                file=sys.stderr, flush=True,
            )
            time.sleep(sleep_s)
    raise last  # defensive; loop above always returns or raises


def _fit(est, *args, _what="fit", **kwargs):
    """Fit `est` (mutates in place, sklearn-style) with transient retry."""
    _retry(est.fit, *args, _what=_what, **kwargs)
    return est


def _predict(est, X, _what="predict"):
    return _retry(est.predict, X, _what=_what)


def _predict_proba(est, X, _what="predict_proba"):
    return _retry(est.predict_proba, X, _what=_what)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, required=True, help="sample size (200/1200/5000)")
    p.add_argument("--sims", type=str, default="1-200", help="e.g. 1-200 or 1,5,7")
    p.add_argument("--scenarios", type=str, default="simple,complex")
    p.add_argument("--cross_fit", type=str, default="nocf,cf")
    p.add_argument("--n_estimators", type=int, default=N_ESTIMATORS,
                   help="TabPFN internal aggregation size (default 8 = out-of-"
                        "the-box; 1 = the manuscript's 'no aggregation' arm)")
    return p.parse_args()


def sim_id_list(spec):
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return sorted({int(s) for s in spec.split(",")})


def make_models(n_estimators=N_ESTIMATORS):
    from tabpfn_client import TabPFNClassifier, TabPFNRegressor
    from tabpfn_client.estimator import ModelVersion

    def clf():
        return TabPFNClassifier.create_default_for_version(
            ModelVersion.V3, n_estimators=n_estimators, random_state=RANDOM_STATE)

    def reg():
        return TabPFNRegressor.create_default_for_version(
            ModelVersion.V3, n_estimators=n_estimators, random_state=RANDOM_STATE)

    return clf, reg


def fit_nuisance(Y, A, X, make_clf, make_reg):
    """Full-sample fits; returns raw pihat (no truncation) + arm-stratified mus."""
    clf = make_clf()
    _fit(clf, X, A, _what="ps.fit")
    pihat = _predict_proba(clf, X, _what="ps.pred")[:, 1]

    mu = {}
    for arm in (0, 1):
        reg = make_reg()
        idx = A == arm
        _fit(reg, X[idx], Y[idx], _what=f"mu{arm}.fit")
        mu[arm] = np.asarray(_predict(reg, X, _what=f"mu{arm}.pred"),
                             dtype=np.float64)

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
        _fit(clf, X[train], A[train], _what=f"ps.fit[f{k}]")
        pihat[test] = _predict_proba(clf, X[test], _what=f"ps.pred[f{k}]")[:, 1]

        for arm, out in ((0, mu0), (1, mu1)):
            idx = train & (A == arm)
            reg = make_reg()
            _fit(reg, X[idx], Y[idx], _what=f"mu{arm}.fit[f{k}]")
            out[test] = np.asarray(
                _predict(reg, X[test], _what=f"mu{arm}.pred[f{k}]"),
                dtype=np.float64)

    return pihat, mu0, mu1


def main():
    args = parse_args()
    data_dir = PROJECT_ROOT / "_data_processed" / f"n{args.n}"
    # n_estimators = 8 -> the default learner ("tabpfn_v3"); any other value
    # gets its own nuisance directory, picked up by script 02's --track tabpfn
    # as learner "tabpfn_v3_ne<k>" (the manuscript uses ne1 at n = 1200 only).
    suffix = "" if args.n_estimators == N_ESTIMATORS else f"_ne{args.n_estimators}"
    out_dir = data_dir / f"tabpfn_v3{suffix}_nuisance"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        sys.exit(f"Missing {data_dir} — run `Rscript 01_generate_dgp_data.R --n {args.n}` first.")

    make_clf, make_reg = make_models(args.n_estimators)
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
