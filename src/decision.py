"""Phase 6 - Decision support: risk × value and budget simulation.

Question: if the budget only covers part of the customer base, does ranking customers by expected loss
(model downgrade probability × current value) retain more revenue than rules
that do not use a model?

Definitions (fixed before looking at the results):
* Rev_i(t): revenue of customer i in the period (t-1, t], scaled to 30 days; only purchase lines not yet cancelled
  at time t are counted (same snapshot-aware rule as Phase 1).
* v_i = Rev_i(t)  - current value, known at t.
* L_i = max(0, Rev_i(t) - Rev_i(t+1))  - revenue actually lost, used for scoring only.
* V(s): mean 30-day revenue of customers currently in state s, estimated on the training periods.
* Downgrade: the next-period state has a lower V than the current state.
* r_i = sum of predicted probabilities of the states with a lower V; EL_i = r_i × v_i.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from config import SNAPSHOT_DATES, SNAPSHOT_ORDER

DATA_START = pd.Timestamp("2009-12-01")


def period_revenue(purchases: pd.DataFrame) -> pd.DataFrame:
    """Revenue of each customer in each period (t-1, t], scaled to 30 days. Columns: CustomerID, snapshot, Rev30, days."""
    rows = []
    prev_end = DATA_START
    for s in SNAPSHOT_ORDER:
        t_end = SNAPSHOT_DATES[s] + pd.Timedelta(days=1)
        m = ((purchases["InvoiceDate"] >= prev_end) & (purchases["InvoiceDate"] < t_end) &
             (purchases["CancelDate"].isna() | (purchases["CancelDate"] >= t_end)))
        days = (t_end - prev_end).days
        r = purchases.loc[m].groupby("CustomerID")["Revenue"].sum().rename("Rev").reset_index()
        r["snapshot"], r["days"] = s, days
        rows.append(r)
        prev_end = t_end
    out = pd.concat(rows, ignore_index=True)
    out["Rev30"] = out["Rev"] * 30 / out["days"]
    return out[["CustomerID", "snapshot", "days", "Rev30"]]


def attach_revenue(ds: pd.DataFrame, rev: pd.DataFrame) -> pd.DataFrame:
    """Attach v (Rev30 at the origin), next-period Rev30 and L to the dataset by (CustomerID, origin)."""
    r = rev.set_index(["CustomerID", "snapshot"])["Rev30"]
    out = ds.copy()
    out["v"] = r.reindex(list(zip(out["CustomerID"], out["origin"]))).fillna(0).values
    out["rev_next"] = r.reindex(list(zip(out["CustomerID"], out["target_snapshot"]))).fillna(0).values
    out["L"] = (out["v"] - out["rev_next"]).clip(lower=0)
    return out


def state_values(train: pd.DataFrame, k: int) -> pd.Series:
    """V(s) = mean 30-day revenue of customers in state s (measured at the target period of the training rows)."""
    return train.groupby(train["Segment_t+1"].astype(int))["rev_next"].mean().reindex(range(k))


def downgrade_mask(V: pd.Series) -> np.ndarray:
    """D[s, s'] = True if moving from s to s' is a downgrade (V(s') < V(s))."""
    v = V.values
    return v[None, :] < v[:, None]


def downgrade_risk(proba: np.ndarray, seg_t: np.ndarray, D: np.ndarray) -> np.ndarray:
    """r_i = sum of the probabilities of the states that are a downgrade from the customer's current state."""
    return (proba * D[seg_t]).sum(axis=1)


def segment_risk(train: pd.DataFrame, D: np.ndarray, k: int) -> pd.Series:
    """Mean downgrade risk by current state (from the training transition matrix) - no model used."""
    st, nx = train["Segment_t"].astype(int).values, train["Segment_t+1"].astype(int).values
    return pd.Series(D[st, nx]).groupby(st).mean().reindex(range(k)).fillna(0)


# ---------------------------------------------------------------- strategy evaluation
def captured_share(df: pd.DataFrame, score: str, frac: float, by: str = "origin") -> float:
    """% of total L captured when the top `frac` customers by `score` are contacted in each origin (each campaign)."""
    got = total = 0.0
    for _, g in df.groupby(by):
        n = int(round(frac * len(g)))
        top = g.sort_values(score, ascending=False, kind="mergesort").head(n)
        got += top["L"].sum(); total += g["L"].sum()
    return got / total


def gain_curve(df: pd.DataFrame, score: str, grid=np.linspace(0, 1, 101), by: str = "origin") -> np.ndarray:
    return np.array([captured_share(df, score, f, by) if f > 0 else 0.0 for f in grid])


def profit_curve(df: pd.DataFrame, score: str, u: float, c: float, m: float, grid=np.linspace(0, 1, 201),
                 by: str = "origin") -> pd.DataFrame:
    """Profit(N) = m·u·ΣL(top N) - c·N, summed over the campaigns (origins)."""
    rows = []
    for f in grid:
        got = n_tot = 0.0
        for _, g in df.groupby(by):
            n = int(round(f * len(g)))
            got += g.sort_values(score, ascending=False, kind="mergesort").head(n)["L"].sum()
            n_tot += n
        rows.append({"frac": f, "contacted": n_tot, "saved_revenue": u * got,
                     "profit": m * u * got - c * n_tot})
    return pd.DataFrame(rows)


def bootstrap_captured_diff(df: pd.DataFrame, s1: str, s2: str, frac: float, n_boot=2000, seed=42) -> np.ndarray:
    """Customer-level bootstrap: difference in % of L captured between two scores (s1 - s2) at level `frac`."""
    rng = np.random.default_rng(seed)
    cust = df["CustomerID"].unique()
    groups = {c: idx for c, idx in df.groupby("CustomerID").indices.items()}
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(cust, len(cust), replace=True)
        idx = np.concatenate([groups[c] for c in pick])
        s = df.iloc[idx].reset_index(drop=True)
        s["_tie"] = rng.random(len(s))          # random tie-breaking between duplicated customers
        out[b] = captured_share(s.sort_values("_tie"), s1, frac) - captured_share(s.sort_values("_tie"), s2, frac)
    return out
