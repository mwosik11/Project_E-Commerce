"""Step 3 — Transitions, tests and prediction dataset (same logic as notebook 03)."""
from common import *
import numpy as np, pandas as pd, pickle
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap
from config import SNAPSHOT_ORDER, N_STATES, STATE_NAMES, STATE_COLORS, TARGET, TUNING_FOLDS, CONFIG_B
from dataset import state_wide, build_dataset
import modeling as M

BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
panel = pd.read_parquet(DATA_K / "panel_with_states_window180.parquet")
wide = state_wide(panel)
pairs = []
for a, b in zip(SNAPSHOT_ORDER[:-1], SNAPSHOT_ORDER[1:]):
    m = wide[a].notna() & wide[b].notna()
    pairs.append(pd.DataFrame({"from": wide.loc[m, a].astype(int).values, "to": wide.loc[m, b].astype(int).values, "transition": f"{a}→{b}"}))
pairs = pd.concat(pairs, ignore_index=True)
counts = pd.crosstab(pairs["from"], pairs["to"]).reindex(index=range(N_STATES), columns=range(N_STATES), fill_value=0)
probs = counts.div(counts.sum(axis=1), axis=0)
stay = (pairs["from"] == pairs["to"]).mean()
print(f"{len(pairs):,} transitions; stay share {stay:.3f}; zero cells {(counts.values == 0).sum()}")
print(probs.round(3).to_string())
pd.concat({"count": counts, "P(to | from)": probs}, axis=1).to_csv(TABLES / "step3_transition_matrix.csv")

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.imshow(probs.values, cmap=BLUES, vmin=0, vmax=1)
for i in range(N_STATES):
    for j in range(N_STATES):
        v = probs.values[i, j]
        ax.text(j, i, f"{v:.0%}\n(n={counts.values[i, j]:,})", ha="center", va="center", fontsize=8.5, color="white" if v > 0.5 else "#1a1a19")
ax.set_xticks(range(N_STATES), [STATE_NAMES[i].replace(" ", "\n", 1) for i in range(N_STATES)], fontsize=8.5)
ax.set_yticks(range(N_STATES), [STATE_NAMES[i] for i in range(N_STATES)], fontsize=8.5)
ax.set_xlabel("state at t+1"); ax.set_ylabel("state at t"); ax.grid(False)
ax.set_title("180-day window RFM: transition probabilities, pooled", loc="left", fontsize=11)
for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig(FIGURES / "step3_transition_matrix.png", dpi=150, bbox_inches="tight")

persist = pd.DataFrame({"overall": pairs.groupby("transition", sort=False).apply(lambda g: (g["from"] == g["to"]).mean()),
    **{STATE_NAMES[s]: pairs[pairs["from"] == s].groupby("transition", sort=False).apply(lambda g: (g["to"] == s).mean()) for s in range(N_STATES)}})
persist.to_csv(TABLES / "step3_persistence_by_transition.csv"); print(persist.round(3).to_string())

homog = []
for s in range(N_STATES):
    tab = pd.crosstab(pairs.loc[pairs["from"] == s, "transition"], pairs.loc[pairs["from"] == s, "to"])
    tab = tab.loc[:, tab.sum() > 0]
    chi2, pval, dof, _ = stats.chi2_contingency(tab); n = tab.values.sum()
    homog.append({"from state": STATE_NAMES[s], "n": n, "chi2": chi2, "dof": dof, "p-value": pval, "Cramér's V": np.sqrt(chi2 / (n * (min(tab.shape) - 1)))})
homog = pd.DataFrame(homog); homog.to_csv(TABLES / "step3_time_homogeneity.csv", index=False); print(homog.round(4).to_string())

ds = build_dataset(panel)
main = ds[ds["history_eligible"] & ds["split"].isin(["train", "test"])]
tr = main[main["split"] == "train"]
rows, G, dof_t = [], 0.0, 0
for s in range(N_STATES):
    tab = pd.crosstab(tr.loc[tr["Segment_t"] == s, "Segment_t-1"], tr.loc[tr["Segment_t"] == s, TARGET])
    tab = tab.loc[tab.sum(axis=1) > 0, tab.sum(axis=0) > 0]
    if min(tab.shape) < 2: continue
    g, p, dof, _ = stats.chi2_contingency(tab, lambda_="log-likelihood")
    rows.append({"Segment_t": STATE_NAMES[s], "n": int(tab.values.sum()), "G": g, "dof": dof, "p-value": p}); G += g; dof_t += dof
markov = pd.DataFrame(rows)
markov.loc[len(markov)] = {"Segment_t": "TOTAL", "n": int(markov["n"].sum()), "G": G, "dof": dof_t, "p-value": stats.chi2.sf(G, dof_t)}
markov.to_csv(TABLES / "step3_markov_order_test_train.csv", index=False); print(markov.round(4).to_string())
print("rows train/test:", (main.split == "train").sum(), (main.split == "test").sum(),
      "| persistence rate train", round(main.loc[main.split == "train", "persisted"].mean(), 3), "test", round(main.loc[main.split == "test", "persisted"].mean(), 3))
print("target distribution test:", main.loc[main.split == "test", TARGET].value_counts(normalize=True).sort_index().round(3).to_dict())
keep = ["CustomerID", "origin", "t", "target_snapshot", "split", "history_eligible", *CONFIG_B, "StateDuration_raw", TARGET, "persisted"]
ds[keep].to_parquet(DATA_K / "prediction_dataset_window180.parquet", index=False)

# tuning folds: states are re-fitted inside each fold on the 180-day window panel (as in Phase 4)
base = pd.read_parquet(DATA / "panel_window180.parquet")
folds = {fd["fold"]: M.fold_data(base, fd)[:2] for fd in TUNING_FOLDS}
pickle.dump(folds, open(DATA_K / "tuning_folds_window180.pkl", "wb"))
print("folds:", {k: (len(v[0]), len(v[1])) for k, v in folds.items()})
