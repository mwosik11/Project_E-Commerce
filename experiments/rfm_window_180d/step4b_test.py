"""Step 4b — Select parameters, final fit on the training origins, single evaluation on the test set (same logic as notebook 04)."""
from common import *
import numpy as np, pandas as pd, pickle, joblib
from matplotlib.colors import LinearSegmentedColormap
from sklearn.inspection import permutation_importance
from config import TARGET, N_STATES, STATE_NAMES, CONFIG_B, RANDOM_STATE
import modeling as M

MODEL_NAMES = ["LogReg", "RandomForest", "XGBoost"]
folds = pickle.load(open(DATA_K / "tuning_folds_window180.pkl", "rb"))
tuning = pd.concat([M.tune(m, c, folds, TUNE_CACHE) for m in MODEL_NAMES for c in ["A", "B"]], ignore_index=True)
summary = (tuning.groupby(["model", "config", "params"]).agg(val_macro_f1=("macro_f1", "mean"),
           fold1=("macro_f1", lambda s: s.iloc[0]), fold2=("macro_f1", lambda s: s.iloc[-1])).reset_index())
best = summary.loc[summary.groupby(["model", "config"])["val_macro_f1"].idxmax()].reset_index(drop=True)
best_params = {(r.model, r.config): M.best_params(tuning[(tuning.model == r.model) & (tuning.config == r.config)])[0] for r in best.itertuples()}
pv = []
for fid, (tr, va) in folds.items():
    pv.append(M.f1_score(va[TARGET].astype(int), va["Segment_t"].astype(int), average="macro"))
best["persistence_val"] = np.mean(pv)
best.to_csv(TABLES / "step4_best_params.csv", index=False)
print(best[["model", "config", "params", "val_macro_f1", "fold1", "fold2"]].to_string(), "\npersistence val", round(np.mean(pv), 4))

ds = pd.read_parquet(DATA_K / "prediction_dataset_window180.parquet")
main = ds[ds["history_eligible"]]
train, test = main[main["split"] == "train"], main[main["split"] == "test"]
y = test[TARGET].astype(int).values; seg = test["Segment_t"].astype(int).values
preds = {(n, "–"): (p, None) for n, p in M.benchmark_predictions(train, test).items()}
final = {}
for m in MODEL_NAMES:
    for c in ["A", "B"]:
        feats = M.CONFIGS[c]
        pipe = M.fit(m, M.make_model(m, feats, best_params[(m, c)]), *M.to_xy(train, feats))
        Xte = M.to_xy(test, feats)[0]
        final[(m, c)] = pipe; preds[(m, c)] = (pipe.predict(Xte), pipe.predict_proba(Xte))
        joblib.dump(pipe, MODELS / f"final_{m}_{c}.joblib")
results = pd.DataFrame([{"model": m, "config": c, **M.metrics(y, p, pr, seg)} for (m, c), (p, pr) in preds.items()])
results.to_csv(TABLES / "step4_test_results.csv", index=False)
print(results[["model", "config", "macro_f1", "weighted_f1", "accuracy", "balanced_accuracy", "macro_auc_ovr", "stayers_accuracy", "changers_accuracy"]].round(4).to_string())
pc = []
for (m, c), (p, _) in preds.items():
    t = M.per_class(y, p); t.insert(0, "state", t.index.map(STATE_NAMES)); t.insert(0, "config", c); t.insert(0, "model", m); pc.append(t)
pc = pd.concat(pc).reset_index(drop=True); pc.to_csv(TABLES / "step4_per_class.csv", index=False)
print(pc.pivot_table(index=["model", "config"], columns="state", values="f1", sort=False).round(3).to_string())

out = test[["CustomerID", "origin", "target_snapshot", "Segment_t", TARGET]].reset_index(drop=True).copy()
for (m, c), (p, pr) in preds.items():
    tag = f"{m}_{c}".replace("–", "bench").replace(" ", "")
    out[f"pred_{tag}"] = p
    if pr is not None:
        for k in range(N_STATES): out[f"proba_{tag}_{k}"] = pr[:, k]
out.to_parquet(DATA_K / "test_predictions_window180.parquet", index=False)

# figures: test Macro-F1 and confusion matrices
CFG = {"A": "#2a78d6", "B": "#eb6834"}
persist = results.loc[results.model == "Persistence", "macro_f1"].item()
fig, ax = plt.subplots(figsize=(9, 3.8)); w = 0.36
for j, c in enumerate(["A", "B"]):
    vals = [results[(results.model == m) & (results.config == c)]["macro_f1"].item() for m in MODEL_NAMES]
    xs = np.arange(3) + (j - 0.5) * w
    ax.bar(xs, vals, width=w - 0.04, color=CFG[c], label=f"Config {c}")
    for x, v in zip(xs, vals): ax.text(x, v + 0.004, f"{v:.3f}", ha="center", fontsize=8.5)
ax.axhline(persist, color="#5c5c57", linestyle="--", linewidth=1.2, label=f"persistence ({persist:.3f})")
ax.set_xticks(range(3), MODEL_NAMES); ax.set_ylim(0.45, 0.66); ax.set_ylabel("test Macro-F1 (S6–S7)")
ax.set_title("180-day window RFM: test Macro-F1 (y-axis starts at 0.45)", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=3); ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(FIGURES / "step4_test_macro_f1.png", dpi=150, bbox_inches="tight")
BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6)); short = [STATE_NAMES[i].split()[0] for i in range(N_STATES)]
for ax, key in zip(axes, [("Persistence", "–"), ("XGBoost", "A"), ("XGBoost", "B")]):
    cm = M.confusion_matrix(y, preds[key][0], labels=M.LABELS); cmn = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(cmn, cmap=BLUES, vmin=0, vmax=1)
    for i in range(N_STATES):
        for j in range(N_STATES):
            ax.text(j, i, f"{cmn[i, j]:.0%}\n{cm[i, j]:,}", ha="center", va="center", fontsize=8, color="white" if cmn[i, j] > 0.5 else "#1a1a19")
    ax.set_xticks(range(N_STATES), short); ax.set_yticks(range(N_STATES), short); ax.set_xlabel("predicted"); ax.set_ylabel("actual state at t+1"); ax.grid(False)
    ax.set_title(key[0] + (f" · Config {key[1]}" if key[1] != "–" else ""), loc="left", fontsize=11)
    for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig(FIGURES / "step4_confusion_matrices.png", dpi=150, bbox_inches="tight")

Xte = M.to_xy(test, CONFIG_B)[0]
r = permutation_importance(final[("XGBoost", "B")], Xte, y, scoring="f1_macro", n_repeats=5, random_state=RANDOM_STATE, n_jobs=1)
imp = pd.Series(r.importances_mean, index=CONFIG_B, name="XGBoost_B").sort_values(ascending=False)
imp.to_csv(TABLES / "step4_permutation_importance.csv"); print(imp.round(4).to_string())
