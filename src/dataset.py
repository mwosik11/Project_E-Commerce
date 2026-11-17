"""Turn a customer-snapshot panel with states into the supervised next-state dataset
(proposal Section 3.5) without look-ahead (Section 3.6).

One row = one customer at origin t, with
  current features at t, Segment_t                        (Config A)
  + Segment_t-1, Segment_t-2, StateDuration at t          (Config B)
  target Segment_t+1.
Every input is taken from snapshot <= t; the target comes from t+1 only.

`build_dataset` takes whatever `Segment` column the panel carries, so Phase 4 can
rebuild the dataset inside each tuning fold after re-fitting the states on that
fold's past data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from config import SNAPSHOT_ORDER, SNAPSHOT_INDEX, CURRENT_FEATURES, DURATION_CAP, TRAIN_ORIGINS, TEST_ORIGINS


def state_wide(panel: pd.DataFrame, col: str = "Segment") -> pd.DataFrame:
    """customers x snapshots matrix of states (NaN = customer not yet in the panel)."""
    return panel.pivot(index="CustomerID", columns="snapshot", values=col).reindex(columns=SNAPSHOT_ORDER)


def run_length(wide: pd.DataFrame) -> pd.DataFrame:
    """Consecutive snapshots (incl. t) a customer has been in its state at t, counted back over observed history."""
    out = pd.DataFrame(np.nan, index=wide.index, columns=wide.columns)
    prev_state, prev_len = None, None
    for s in wide.columns:
        cur = wide[s]
        if prev_state is None:
            ln = cur.notna().astype(float)
        else:
            same = cur.notna() & (cur == prev_state)
            ln = np.where(same, prev_len + 1, np.where(cur.notna(), 1.0, np.nan))
            ln = pd.Series(ln, index=wide.index)
        out[s] = ln.where(cur.notna())
        prev_state, prev_len = cur, out[s].fillna(0)
    return out


def build_dataset(panel: pd.DataFrame, duration_cap: int = DURATION_CAP) -> pd.DataFrame:
    wide = state_wide(panel)
    runs = run_length(wide)
    rows = []
    for i, s in enumerate(SNAPSHOT_ORDER[:-1]):
        nxt = SNAPSHOT_ORDER[i + 1]
        cur = panel[panel["snapshot"] == s].set_index("CustomerID")
        d = cur[["snapshot", "t"] + CURRENT_FEATURES].copy()
        d["Segment_t"] = cur["Segment"]
        d["Segment_t-1"] = wide[SNAPSHOT_ORDER[i - 1]].reindex(d.index) if i >= 1 else np.nan
        d["Segment_t-2"] = wide[SNAPSHOT_ORDER[i - 2]].reindex(d.index) if i >= 2 else np.nan
        d["StateDuration_raw"] = runs[s].reindex(d.index)
        d["StateDuration"] = d["StateDuration_raw"].clip(upper=duration_cap)
        d["Segment_t+1"] = wide[nxt].reindex(d.index)
        d["target_snapshot"] = nxt
        rows.append(d.reset_index())
    ds = pd.concat(rows, ignore_index=True).rename(columns={"snapshot": "origin"})
    ds = ds[ds["Segment_t+1"].notna()].copy()                       # customers whose all purchases were later cancelled
    ds["history_eligible"] = ds["Segment_t-2"].notna()
    ds["split"] = np.select([ds["origin"].isin(TRAIN_ORIGINS), ds["origin"].isin(TEST_ORIGINS)], ["train", "test"], "none")
    for c in ["Segment_t", "Segment_t-1", "Segment_t-2", "Segment_t+1"]:
        ds[c] = ds[c].astype("Int64")
    ds["persisted"] = ds["Segment_t+1"] == ds["Segment_t"]
    return ds.reset_index(drop=True)
