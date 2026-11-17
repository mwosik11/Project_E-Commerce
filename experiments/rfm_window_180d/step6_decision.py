"""Step 6 — Risk × value and budget simulation (same definitions, assumptions and parameters as notebook 06)."""
from common import *
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, r2_score, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor
from config import INTERIM, N_STATES, STATE_NAMES, CURRENT_FEATURES, RANDOM_STATE
from decision import (period_revenue, attach_revenue, state_values, downgrade_mask, downgrade_risk, segment_risk,
                      captured_share, gain_curve, profit_curve, bootstrap_captured_diff)

rev = period_revenue(pd.read_parquet(INTERIM / "transactions_clean.parquet"))
ds = attach_revenue(pd.read_parquet(DATA_K / "prediction_dataset_window180.parquet"), rev)
train_all = ds[ds["split"] == "train"]
V = state_values(train_all, N_STATES); D = downgrade_mask(V); segr = segment_risk(train_all, D, N_STATES)
st = pd.DataFrame({"state": [STATE_NAMES[s] for s in range(N_STATES)], "V_revenue_per_30d": V.values, "downgrade_rate_train": segr.values})
st.to_csv(TABLES / "step6_state_values.csv", index=False); print(st.round(3).to_string())

te = attach_revenue(pd.read_parquet(DATA_K / "test_predictions_window180.parquet"), rev)
seg, nxt = te["Segment_t"].astype(int).values, te["Segment_t+1"].astype(int).values
te["down"] = D[seg, nxt]; te["stop"] = (te["v"] > 0) & (te["rev_next"] == 0)
te["r_raw"] = downgrade_risk(te[[f"proba_XGBoost_B_{k}" for k in range(N_STATES)]].values, seg, D)
print("downgrade AUC", round(roc_auc_score(te.down, te.r_raw), 3), "Brier", round(brier_score_loss(te.down, te.r_raw), 4), "| downgrade rate", round(te.down.mean(), 3))
tot = te["L"].sum()
decomp = pd.Series({"stayed in same state": te.loc[nxt == seg, "L"].sum() / tot, "downgraded": te.loc[te.down, "L"].sum() / tot,
                    "upgraded": te.loc[(nxt != seg) & ~te.down, "L"].sum() / tot, "of which: stopped buying": te.loc[te.stop, "L"].sum() / tot})
decomp.to_frame("share_of_total_L").to_csv(TABLES / "step6_loss_decomposition.csv"); print(decomp.round(3).to_dict(), "total L", round(tot))
te["L_stop"] = np.where(te.stop, te.v, 0.0); te["L_down"] = np.where(te.down, te.L, 0.0)

# two direct models — same configuration and the same fixed parameters as notebook 06
dh = ds[ds["history_eligible"]].copy(); dh["v_feat"] = dh["v"]
F = CURRENT_FEATURES + ["v_feat", "Segment_t", "Segment_t-1", "Segment_t-2", "StateDuration"]
X = lambda d: d[F].astype(float)
trh = dh[dh.split == "train"]; trp = trh[trh.v > 0]
clf = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=4, eval_metric="logloss").fit(X(trp), (trp.rev_next == 0).astype(int))
reg = XGBRegressor(objective="reg:tweedie", tweedie_variance_power=1.5, n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=4).fit(X(trh), trh.rev_next)
te = te.merge(dh.loc[dh.split == "test", ["CustomerID", "origin"] + [c for c in F if c != "Segment_t"]], on=["CustomerID", "origin"], how="left")
act = te.v > 0
te["p_stop"] = np.where(act, clf.predict_proba(X(te))[:, 1], 0.0); te["rev_hat"] = reg.predict(X(te))
direct = pd.Series({"AUC P(stop)": roc_auc_score(te.loc[act, "stop"], te.loc[act, "p_stop"]), "R2 regression": r2_score(te.rev_next, te.rev_hat),
                    "MAE regression": mean_absolute_error(te.rev_next, te.rev_hat), "R2 naive": r2_score(te.rev_next, te.v)})
direct.to_frame("value").to_csv(TABLES / "step6_direct_models.csv"); print(direct.round(3).to_dict())

te["EL_state"] = te.r_raw * te.v; te["SEG_EL"] = segr.reindex(seg).values * te.v
te["EL_stop"] = te.p_stop * te.v; te["EL_reg"] = (te.v - te.rev_hat).clip(lower=0)
te["random"] = np.random.default_rng(RANDOM_STATE).random(len(te))
LAB = {"random": "Random", "v": "Value only", "SEG_EL": "Segment risk × value", "EL_state": "State-model risk × value",
       "EL_stop": "Stop-model risk × value", "EL_reg": "Regression: expected decline", "L": "Oracle"}
rows = []
for out in ["L", "L_stop", "L_down"]:
    tmp = te.assign(L=te[out])
    for s in LAB: rows.append({"outcome": out, "strategy": LAB[s], **{f"top {int(f*100)}%": captured_share(tmp, s, f) for f in (0.05, 0.10, 0.20)}})
cap = pd.DataFrame(rows); cap.to_csv(TABLES / "step6_captured_share.csv", index=False)
print(cap.pivot_table(index="strategy", columns="outcome", values="top 10%", sort=False).round(3).to_string())
ci = []
for out, s in [("L_stop", "EL_stop"), ("L_stop", "EL_state"), ("L_stop", "SEG_EL"), ("L", "EL_stop"), ("L", "EL_state"), ("L", "EL_reg")]:
    tmp = te.assign(L=te[out]); pt = captured_share(tmp, s, 0.10) - captured_share(tmp, "v", 0.10)
    dr = bootstrap_captured_diff(tmp, s, "v", 0.10, n_boot=1000, seed=RANDOM_STATE)
    ci.append({"outcome": out, "strategy": LAB[s], "diff_vs_value_top10": pt, "ci_low": np.percentile(dr, 2.5), "ci_high": np.percentile(dr, 97.5)})
ci = pd.DataFrame(ci); ci.to_csv(TABLES / "step6_bootstrap_ci.csv", index=False); print(ci.round(3).to_string())
sim = te.assign(L=te["L_stop"] * 3); M_ = 0.25; scen = []
for u in [0.05, 0.10, 0.20]:
    for c in [2, 10, 25]:
        for s in ["v", "SEG_EL", "EL_state", "EL_stop"]:
            pc = profit_curve(sim, s, u=u, c=c, m=M_); b = pc.loc[pc.profit.idxmax()]
            scen.append({"u": u, "c": c, "strategy": LAB[s], "best_frac": b.frac, "best_profit": b.profit})
scen = pd.DataFrame(scen); scen.to_csv(TABLES / "step6_budget_scenarios.csv", index=False)
print(scen.pivot_table(index=["u", "c"], columns="strategy", values="best_profit", sort=False).round(0).to_string())
be = []
n10 = sum(int(round(0.10 * len(g))) for _, g in sim.groupby("origin"))
for s in ["v", "EL_state", "EL_stop"]:
    got = captured_share(sim, s, 0.10) * sim.L.sum()
    for c in [2, 10, 25]: be.append({"strategy": LAB[s], "c": c, "break_even_u": c * n10 / (M_ * got)})
be = pd.DataFrame(be); be.to_csv(TABLES / "step6_breakeven.csv", index=False)
print(be.pivot_table(index="strategy", columns="c", values="break_even_u", sort=False).round(3).to_string())

grid = np.linspace(0, 1, 101)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
show = [("random", "#9a9a92", "--"), ("v", "#2a78d6", "-"), ("SEG_EL", "#1baf7a", "-"), ("EL_state", "#eda100", "-"), ("EL_stop", "#eb6834", "-"), ("L", "#1a1a19", ":")]
for ax, (out, title) in zip(axes, [("L_stop", "Revenue lost because customers stop buying (L_stop)"), ("L", "Total revenue decline (L)")]):
    tmp = te.assign(L=te[out])
    for s, col, ls in show: ax.plot(grid * 100, gain_curve(tmp, s, grid) * 100, color=col, ls=ls, label=LAB[s])
    ax.axvline(10, color="#c9c8c2", lw=1); ax.set_xlim(0, 50); ax.set_title(title, loc="left", fontsize=10); ax.set_xlabel("% of customers contacted (per campaign)")
axes[0].set_ylabel("% of lost revenue captured"); axes[1].legend(loc="lower right", fontsize=8, frameon=False)
fig.suptitle("180-day window RFM: gain curves", x=0.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(FIGURES / "step6_gain_curves.png", dpi=150, bbox_inches="tight")
