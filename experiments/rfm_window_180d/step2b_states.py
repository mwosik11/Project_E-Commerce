"""Step 2b — Choose k with exactly the rule of the main design, fit and freeze the states, describe the states."""
from common import *
import numpy as np, pandas as pd, joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from config import TRAIN_ORIGINS, SNAPSHOT_ORDER, RANDOM_STATE
from states import fit_state_model
from k_selection import decide, elbow_table, PS_THRESHOLD
from step2a_kselect import RUNS, run_window, panel

# ---- choose k: Elbow (Kneedle) by majority over 6 runs — same rule as notebook 02
k_table = run_window(TRAIN_ORIGINS, RANDOM_STATE, ("internal", "ps", "boot"))
k_table = k_table.join(elbow_table(k_table["inertia"]).drop(columns="inertia"))
rob = []
for snaps, seed in RUNS:
    t = run_window(snaps, seed); d = decide(t); t3 = t[t.index >= 3]
    rob.append({"window": f"{snaps[0]}–{snaps[-1]}", "seed": seed, **d,
                "best Silhouette (k≥3)": int(t3["silhouette"].idxmax()), "best DBI (k≥3)": int(t3["davies_bouldin"].idxmin()),
                "best CH (k≥3)": int(t3["calinski_harabasz"].idxmax())})
rob = pd.DataFrame(rob)
K = WINDOW_K or int(rob["k_elbow_kneedle"].value_counts().idxmax())   # WINDOW_K: sensitivity analysis with another k
print("Kneedle votes:", rob["k_elbow_kneedle"].value_counts().to_dict(), "| 2nd-diff votes:", rob["k_elbow_2nd_diff"].value_counts().to_dict(),
      "| best Silhouette k>=3:", rob["best Silhouette (k≥3)"].value_counts().to_dict(), "-> K =", K)
k_table.to_csv(TABLES / "step2_k_selection.csv"); rob.to_csv(TABLES / "step2_k_selection_robustness.csv", index=False)

# ---- k-selection figure (same layout as the main design)
ks = k_table.index.values; blue, grey = "#2a78d6", "#5c5c57"
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
panels = [("inertia", None, "Elbow: inertia (WCSS)  — primary"), ("drop_pct", None, "Elbow: % inertia reduction vs. k−1"),
          ("silhouette", None, "Silhouette (higher = better)"), ("davies_bouldin", None, "Davies-Bouldin (lower = better)"),
          ("calinski_harabasz", None, "Calinski-Harabasz (higher = better)"), ("prediction_strength", "ps_sd", "Prediction strength (± sd)  — gate PS ≥ 0.8")]
for ax, (col, err, title) in zip(axes.ravel(), panels):
    y = k_table[col].values
    if err: ax.errorbar(ks, y, yerr=k_table[err].values, color=blue, marker="o", markersize=6, capsize=3)
    elif col == "drop_pct": ax.bar(ks, np.nan_to_num(y), color=blue, width=0.6)
    else: ax.plot(ks, y, color=blue, marker="o", markersize=6)
    if col != "drop_pct":
        ax.plot([K], [k_table.loc[K, col]], marker="o", markersize=11, markerfacecolor="none", markeredgecolor="#1a1a19", markeredgewidth=1.5)
    if col == "prediction_strength": ax.axhline(PS_THRESHOLD, color=grey, linestyle="--", linewidth=1)
    ax.set_title(title, loc="left", fontsize=10); ax.set_xticks(ks); ax.set_xlabel("k")
fig.suptitle(f"180-day window RFM: selecting the number of states on S1–S5 (circled: k = {K})", x=0.01, ha="left", fontsize=12)
fig.tight_layout(); fig.savefig(FIGURES / "step2_k_selection.png", dpi=150, bbox_inches="tight")

# ---- fit and freeze the states on S1–S5
sm = fit_state_model(panel, TRAIN_ORIGINS, k=K)
joblib.dump(sm, MODELS / "state_model_window180.joblib")
pnl = panel.copy(); pnl["Segment"] = sm.predict(pnl)
tr = pnl[pnl.snapshot.isin(TRAIN_ORIGINS)]
prof = tr.groupby("Segment").agg(share=("CustomerID", lambda s: len(s) / len(tr)), Recency=("Recency", "median"),
                                 F_180d=("Frequency", "median"), M_180d=("Monetary", "median"),
                                 F_total=("FrequencyTotal", "median"), M_total=("MonetaryTotal", "median"),
                                 zero_F_share=("Frequency", lambda s: (s == 0).mean()), Tenure=("Tenure", "median"),
                                 active_90d=("RecentOrders", lambda s: (s > 0).mean()))
print(prof.round(2).to_string()); print(sm.centers_z.round(2).to_string())
prof.to_csv(TABLES / "step2_state_profiles.csv"); sm.centers_z.to_csv(TABLES / "step2_centroids_z.csv")
pnl.to_parquet(DATA_K / "panel_with_states_window180.parquet", index=False)

share = pnl.groupby("snapshot")["Segment"].value_counts(normalize=True).unstack().reindex(SNAPSHOT_ORDER).fillna(0)
share.to_csv(TABLES / "step2_state_share_by_snapshot.csv")
drift = []
for s in SNAPSHOT_ORDER:
    r = pnl[pnl.snapshot == s]; Zs = sm.transform(r)
    refit = KMeans(n_clusters=K, n_init=20, random_state=RANDOM_STATE).fit_predict(Zs)
    drift.append({"snapshot": s, "split": "train" if s in TRAIN_ORIGINS else "out-of-sample",
                  "silhouette (frozen)": silhouette_score(Zs, r["Segment"], sample_size=min(4000, len(Zs)), random_state=0),
                  "ARI frozen vs refit": adjusted_rand_score(r["Segment"], refit),
                  "mean dist. to own centroid": sm.distances(r)[np.arange(len(r)), r["Segment"].values].mean()})
drift = pd.DataFrame(drift); drift.to_csv(TABLES / "step2_state_drift_check.csv", index=False)
print(share.round(3).to_string()); print(drift.round(3).to_string())
