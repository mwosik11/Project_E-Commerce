"""Paired, customer-level bootstrap for H1/H2 (proposal Section 3.7).

A customer can appear in both test origins (S6 and S7), so rows are not independent.
Each bootstrap replicate resamples CUSTOMERS with replacement and keeps all of their
rows; every model is evaluated on the SAME resampled customers (paired design), so
differences such as Delta_c = MacroF1(B,c) - MacroF1(A,c) are computed within a replicate.

Fast implementation: per-customer confusion counts (n_customers x K*K) are combined
with the bootstrap multiplicity matrix, so every metric derived from the confusion
matrix is available for all replicates at once.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def customer_confusions(customers: np.ndarray, y: np.ndarray, pred: np.ndarray, k: int):
    """Return (unique customer ids, matrix n_customers x k*k of confusion counts)."""
    uniq, inv = np.unique(customers, return_inverse=True)
    out = np.zeros((len(uniq), k * k))
    np.add.at(out, (inv, y * k + pred), 1)
    return uniq, out


def multiplicities(n_customers: int, n_boot: int, seed: int) -> np.ndarray:
    """n_boot x n_customers matrix: how often each customer is drawn in each replicate."""
    rng = np.random.default_rng(seed)
    W = np.empty((n_boot, n_customers), dtype=np.float32)
    for b in range(n_boot):
        W[b] = np.bincount(rng.integers(0, n_customers, n_customers), minlength=n_customers)
    return W


def cm_metrics(cm: np.ndarray, k: int) -> dict:
    """Metrics from confusion counts; cm has shape (..., k*k), rows = actual, cols = predicted."""
    cm = cm.reshape(cm.shape[:-1] + (k, k))
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    actual = cm.sum(axis=-1)
    predicted = cm.sum(axis=-2)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(predicted > 0, tp / predicted, 0.0)
        rec = np.where(actual > 0, tp / actual, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    total = actual.sum(axis=-1)
    return {"macro_f1": f1.mean(axis=-1),
            "weighted_f1": (f1 * actual).sum(axis=-1) / total,
            "accuracy": tp.sum(axis=-1) / total,
            "balanced_accuracy": np.nanmean(np.where(actual > 0, rec, np.nan), axis=-1)}


def summarize(point: float, draws: np.ndarray, alternative: str = "two-sided") -> dict:
    """Percentile 95 % CI and bootstrap p-value for H0: value = 0."""
    lo, hi = np.percentile(draws, [2.5, 97.5])
    if alternative == "greater":            # H1: value > 0
        p = (np.sum(draws <= 0) + 1) / (len(draws) + 1)
    else:
        p = min(1.0, 2 * min((np.sum(draws <= 0) + 1) / (len(draws) + 1), (np.sum(draws >= 0) + 1) / (len(draws) + 1)))
    return {"estimate": point, "ci_low": lo, "ci_high": hi, "boot_se": draws.std(ddof=1), "p_value": p}


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i]); adj[i] = min(1.0, running)
    return adj
