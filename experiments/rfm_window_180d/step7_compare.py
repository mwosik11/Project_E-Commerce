"""Step 7 — Compare the main design (cumulative RFM) with the 180-day window design. All figures are read from result tables."""
from common import *
import numpy as np, pandas as pd
from config import SNAPSHOT_ORDER, STATE_NAMES, STATE_COLORS

MT, ET = MAIN_TABLES, TABLES
def rd(d, f, **k): return pd.read_csv(d / f, **k)

# --- read
m_share = rd(MT, "phase2_state_distribution.csv", header=[0, 1], index_col=0)["share"]; m_share.columns = [int(c.split()[0]) if " " in c else int(c) for c in m_share.columns]
e_share = rd(ET, "step2_state_share_by_snapshot.csv", index_col=0); e_share.columns = [int(c) for c in e_share.columns]
m_drift, e_drift = rd(MT, "phase2_state_drift_check.csv"), rd(ET, "step2_state_drift_check.csv")
m_rob, e_rob = rd(MT, "phase2_k_selection_robustness.csv"), rd(ET, "step2_k_selection_robustness.csv")
m_tm = rd(MT, "phase3_transition_matrix.csv", header=[0, 1], index_col=0)["count"]; e_tm = rd(ET, "step3_transition_matrix.csv", header=[0, 1], index_col=0)["count"]
m_mk, e_mk = rd(MT, "phase3_markov_order_test_train.csv"), rd(ET, "step3_markov_order_test_train.csv")
m_res, e_res = rd(MT, "phase4_test_results.csv"), rd(ET, "step4_test_results.csv")
m_vp, e_vp = rd(MT, "phase5_vs_persistence.csv"), rd(ET, "step5_vs_persistence.csv")
m_h1, e_h1 = rd(MT, "phase5_h1.csv"), rd(ET, "step5_h1.csv"); m_h2, e_h2 = rd(MT, "phase5_h2.csv"), rd(ET, "step5_h2.csv")
m_dec = rd(MT, "phase6_loss_decomposition.csv", index_col=0)["share_of_total_L"]; e_dec = rd(ET, "step6_loss_decomposition.csv", index_col=0)["share_of_total_L"]
m_cap, e_cap = rd(MT, "phase6_captured_share.csv"), rd(ET, "step6_captured_share.csv")
m_sc, e_sc = rd(MT, "phase6_budget_scenarios.csv"), rd(ET, "step6_budget_scenarios.csv")

def stay(tm): v = tm.values; return np.trace(v) / v.sum()
def zeros(tm): return int((tm.values == 0).sum())
def oos(df, col): x = df.loc[df["split"] == "out-of-sample", col]; return f"{x.min():.2f}–{x.max():.2f}"
def votes(rob, col): vc = rob[col].value_counts(); return ", ".join(f"k={k}: {v}/6" for k, v in vc.items())
def best(res):
    r = res[~res.model.isin(["Persistence", "Majority class"])].sort_values("macro_f1", ascending=False).iloc[0]; return r
def pers(res, col="macro_f1"): return res.loc[res.model == "Persistence", col].item()
def vp_best(vp): r = vp.sort_values("Macro-F1 − persistence", ascending=False).iloc[0]; return f"{r['Macro-F1 − persistence']:+.3f} [{r.ci_low:+.3f}; {r.ci_high:+.3f}] ({r.model} {r.config}); p Holm {r['p Holm']:.3f}"
def n_sig(vp): return int(((vp["p Holm"] < 0.05) & (vp["Macro-F1 − persistence"] > 0)).sum())
def changers(res): x = res[~res.model.isin(["Persistence", "Majority class"])]["changers_accuracy"]; return f"{x.min():.0%}–{x.max():.0%}"
def h1txt(h): s = h[h["p Holm"] < 0.05]["classifier"].tolist(); return ("supported for " + ", ".join(s)) if s else "not supported"
def h2txt(h): s = h[h["p Holm"] < 0.05]["comparison"].tolist(); return ("supported: " + ", ".join(s)) if s else "not supported"
def cap(c, strat, out): return c[(c.strategy.str.contains(strat)) & (c.outcome == out)]["top 10%"].iloc[0]
def prof(s, strat, u, c): return s[(s.u == u) & (s.c == c) & (s.strategy.str.contains(strat))]["best_profit"].iloc[0]
k_m, k_e = m_cap, e_cap
rows = [
    ("States", "Number of states k (rule: Elbow/Kneedle majority)", "4", "4"),
    ("States", "Kneedle votes (6 runs)", votes(m_rob, "k_elbow_kneedle"), votes(e_rob, "k_elbow_kneedle")),
    ("States", "Second-difference votes", votes(m_rob, "k_elbow_2nd_diff"), votes(e_rob, "k_elbow_2nd_diff")),
    ("States", "Best Silhouette among k ≥ 3", votes(m_rob, "best Silhouette (k≥3)"), votes(e_rob, "best Silhouette (k≥3)")),
    ("States", "Active High-Value share S1 → S8", f"{m_share.loc['S1', 3]:.0%} → {m_share.loc['S8', 3]:.0%}", f"{e_share.loc['S1', 3]:.0%} → {e_share.loc['S8', 3]:.0%}"),
    ("States", "Dormant Low-Value share S1 → S8", f"{m_share.loc['S1', 0]:.0%} → {m_share.loc['S8', 0]:.0%}", f"{e_share.loc['S1', 0]:.0%} → {e_share.loc['S8', 0]:.0%}"),
    ("States", "Silhouette of frozen states, S6–S8", oos(m_drift, "silhouette (frozen)"), oos(e_drift, "silhouette (frozen)")),
    ("States", "ARI frozen vs re-fit, S6–S8", oos(m_drift, "ARI frozen vs refit"), oos(e_drift, "ARI frozen vs refit")),
    ("Transitions", "Customers keeping their state per quarter", f"{stay(m_tm):.0%}", f"{stay(e_tm):.0%}"),
    ("Transitions", "Zero cells in transition matrix", str(zeros(m_tm)), str(zeros(e_tm))),
    ("Transitions", "G-test 1st vs 2nd order (total)", f"G = {m_mk.G.iloc[-1]:.0f}, df = {int(m_mk.dof.iloc[-1])}", f"G = {e_mk.G.iloc[-1]:.0f}, df = {int(e_mk.dof.iloc[-1])}"),
    ("Prediction", "Persistence: test Macro-F1 / accuracy", f"{pers(m_res):.3f} / {pers(m_res, 'accuracy'):.3f}", f"{pers(e_res):.3f} / {pers(e_res, 'accuracy'):.3f}"),
    ("Prediction", "Best classifier: test Macro-F1", f"{best(m_res).macro_f1:.3f} ({best(m_res).model} {best(m_res).config})", f"{best(e_res).macro_f1:.3f} ({best(e_res).model} {best(e_res).config})"),
    ("Prediction", "Best − persistence, 95% CI", vp_best(m_vp), vp_best(e_vp)),
    ("Prediction", "Classifiers significantly better than persistence", f"{n_sig(m_vp)} of 6", f"{n_sig(e_vp)} of 6"),
    ("Prediction", "State changers predicted correctly", changers(m_res), changers(e_res)),
    ("Hypotheses", "H1 (history helps, Holm)", h1txt(m_h1), h1txt(e_h1)),
    ("Hypotheses", "H2 (gain differs, Holm)", h2txt(m_h2), h2txt(e_h2)),
    ("Decision", "Lost revenue from customers who downgrade", f"{m_dec.iloc[1]:.0%}", f"{e_dec.loc['downgraded']:.0%}"),
    ("Decision", "Lost revenue from customers keeping their state", f"{m_dec.iloc[0]:.0%}", f"{e_dec.loc['stayed in same state']:.0%}"),
    ("Decision", "L_stop captured at top 10%: value only", f"{cap(m_cap, 'Value only', 'L_stop'):.1%}", f"{cap(e_cap, 'Value only', 'L_stop'):.1%}"),
    ("Decision", "L_stop captured at top 10%: state-model risk × value", f"{cap(m_cap, 'State-model', 'L_stop'):.1%}", f"{cap(e_cap, 'State-model', 'L_stop'):.1%}"),
    ("Decision", "L_stop captured at top 10%: stop-model risk × value", f"{cap(m_cap, 'Stop-model', 'L_stop'):.1%}", f"{cap(e_cap, 'Stop-model', 'L_stop'):.1%}"),
    ("Decision", "Max profit u = 10%, c = £10: state model / stop model", f"£{prof(m_sc, 'State-model', 0.1, 10):,.0f} / £{prof(m_sc, 'Stop-model', 0.1, 10):,.0f}", f"£{prof(e_sc, 'State-model', 0.1, 10):,.0f} / £{prof(e_sc, 'Stop-model', 0.1, 10):,.0f}"),
]
comp = pd.DataFrame(rows, columns=["area", "measure", "cumulative RFM (main)", "180-day window RFM"])
comp.to_csv(TABLES / "comparison_main_vs_window180.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70); print(comp.to_string())

# --- comparison figure: state shares over time under the two designs
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
for ax, sh, title in [(axes[0], m_share, "Cumulative RFM (main design)"), (axes[1], e_share, "180-day window RFM")]:
    bottom = np.zeros(len(SNAPSHOT_ORDER))
    for s in range(4):
        v = sh.reindex(SNAPSHOT_ORDER)[s].values
        ax.bar(SNAPSHOT_ORDER, v, bottom=bottom, color=STATE_COLORS[s], width=0.7, label=STATE_NAMES[s])
        for i, (b, val) in enumerate(zip(bottom, v)):
            if val > 0.06: ax.text(i, b + val / 2, f"{val:.0%}", ha="center", va="center", fontsize=7.5, color="white")
        bottom += v
    ax.axvline(4.5, color="#5c5c57", ls="--", lw=1); ax.set_title(title, loc="left", fontsize=11); ax.grid(axis="x", visible=False)
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
axes[1].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
fig.suptitle("Share of customers in each state by snapshot (right of dashed line: frozen states, out of sample)", x=0.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(FIGURES / "compare_state_shares.png", dpi=150, bbox_inches="tight")

# --- comparison figure: Macro-F1 gain over persistence of the 6 models under the two designs
fig, ax = plt.subplots(figsize=(9, 4))
lab = [f"{m} · {c}" for m, c in zip(m_vp.model, m_vp.config)]; ys = np.arange(len(lab))[::-1]
for vp, off, col, name in [(m_vp, 0.15, "#2a78d6", "Cumulative RFM (main)"), (e_vp, -0.15, "#eb6834", "180-day window RFM")]:
    e = vp["Macro-F1 − persistence"].values
    ax.errorbar(e, ys + off, xerr=[e - vp.ci_low.values, vp.ci_high.values - e], fmt="o", color=col, capsize=3, label=name)
ax.axvline(0, color="#5c5c57", ls="--", lw=1); ax.set_yticks(ys, lab); ax.grid(axis="y", visible=False)
ax.set_xlabel("Macro-F1(model) − Macro-F1(persistence), 95% bootstrap CI"); ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.set_title("Gain over the persistence benchmark under each state definition", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(FIGURES / "compare_vs_persistence.png", dpi=150, bbox_inches="tight")
print("saved")
