"""Step 1 — Customer × snapshot panel with Frequency and Monetary computed over a 180-day window."""
from common import *
import numpy as np, pandas as pd
from config import SNAPSHOT_DATES, SNAPSHOT_ORDER, INTERIM, PROCESSED

pur = pd.read_parquet(INTERIM / "transactions_clean.parquet")          # clean transactions with cancellation dates (Phase 1)
base = pd.read_parquet(PROCESSED / "customer_snapshot_panel.parquet")   # main panel (same customers, same snapshots)

parts = []
for s, t in SNAPSHOT_DATES.items():
    t_end = t + pd.Timedelta(days=1)
    # same snapshot-aware cancellation rule as Phase 1: only drop lines cancelled before t_end
    valid = pur[(pur["InvoiceDate"] < t_end) & (pur["CancelDate"].isna() | (pur["CancelDate"] >= t_end))]
    win = valid[valid["InvoiceDate"] >= t_end - pd.Timedelta(days=WINDOW_DAYS)]
    g = win.groupby("CustomerID")
    d = pd.DataFrame({"Frequency_w": g["InvoiceNo"].nunique(), "Monetary_w": g["Revenue"].sum()}).reset_index()
    d["snapshot"] = s
    parts.append(d)
w = pd.concat(parts, ignore_index=True)

panel = base.merge(w, on=["CustomerID", "snapshot"], how="left")
panel["FrequencyTotal"], panel["MonetaryTotal"] = panel["Frequency"], panel["Monetary"]   # keep the cumulative values for reference
panel["Frequency"] = panel["Frequency_w"].fillna(0).astype(int)      # customers with no purchase in 180 days -> 0
panel["Monetary"] = panel["Monetary_w"].fillna(0.0)
panel = panel.drop(columns=["Frequency_w", "Monetary_w"])
assert len(panel) == len(base)
panel.to_parquet(DATA / "panel_window180.parquet", index=False)

# length of data available inside the window at each snapshot (data begin on 01/12/2009)
start = pd.Timestamp("2009-12-01")
summ = panel.groupby("snapshot").agg(customers=("CustomerID", "size"),
                                     zero_F_share=("Frequency", lambda s: (s == 0).mean()),
                                     median_F_window=("Frequency", "median"), median_F_total=("FrequencyTotal", "median"),
                                     median_M_window=("Monetary", "median"), median_M_total=("MonetaryTotal", "median")).reindex(SNAPSHOT_ORDER)
summ["days_of_data_in_window"] = [min(WINDOW_DAYS, (SNAPSHOT_DATES[s] + pd.Timedelta(days=1) - start).days) for s in SNAPSHOT_ORDER]
summ.to_csv(TABLES / "step1_panel_summary.csv")
skew = panel[panel.snapshot.isin(["S1", "S2", "S3", "S4", "S5"])][["Recency", "Frequency", "Monetary"]].skew()
print(summ.round(2).to_string()); print("train skew:", skew.round(2).to_dict())
