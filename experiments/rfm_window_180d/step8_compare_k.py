"""Step 8 — Compare three designs: cumulative RFM (main, k=4), 180-day window k=4 and 180-day window k=3."""
from common import EXP, ROOT
import numpy as np, pandas as pd
D = {"cumulative RFM, k = 4 (main)": ROOT / "outputs" / "tables", "180-day window, k = 4": EXP / "outputs" / "tables",
     "180-day window, k = 3": EXP / "outputs_k3" / "tables"}
F = {"main": dict(drift="phase2_state_drift_check.csv", tm="phase3_transition_matrix.csv", res="phase4_test_results.csv",
                  vp="phase5_vs_persistence.csv", h1="phase5_h1.csv", h2="phase5_h2.csv", dec="phase6_loss_decomposition.csv", cap="phase6_captured_share.csv"),
     "exp": dict(drift="step2_state_drift_check.csv", tm="step3_transition_matrix.csv", res="step4_test_results.csv",
                 vp="step5_vs_persistence.csv", h1="step5_h1.csv", h2="step5_h2.csv", dec="step6_loss_decomposition.csv", cap="step6_captured_share.csv")}
cols = {}
for name, d in D.items():
    f = F["main" if "main" in name else "exp"]
    dr = pd.read_csv(d / f["drift"]); oos = dr[dr.split == "out-of-sample"]
    tm = pd.read_csv(d / f["tm"], header=[0, 1], index_col=0)["count"].values
    res = pd.read_csv(d / f["res"]); vp = pd.read_csv(d / f["vp"]); h1 = pd.read_csv(d / f["h1"]); h2 = pd.read_csv(d / f["h2"])
    dec = pd.read_csv(d / f["dec"], index_col=0).iloc[:, 0]; cap = pd.read_csv(d / f["cap"])
    pers = res.loc[res.model == "Persistence"].iloc[0]
    best = vp.sort_values("Macro-F1 − persistence", ascending=False).iloc[0]
    mod = res[~res.model.isin(["Persistence", "Majority class"])]
    c10 = lambda key: cap[(cap.outcome == "L_stop") & cap.strategy.str.contains(key)]["top 10%"].iloc[0]
    h1s = h1[h1["p Holm"] < 0.05]["classifier"].tolist(); h2s = h2[h2["p Holm"] < 0.05]["comparison"].tolist()
    cols[name] = {
        "Silhouette of frozen states, S6–S8": f"{oos['silhouette (frozen)'].min():.2f}–{oos['silhouette (frozen)'].max():.2f}",
        "ARI frozen vs re-fit, S6–S8": f"{oos['ARI frozen vs refit'].min():.2f}–{oos['ARI frozen vs refit'].max():.2f}",
        "Customers keeping their state per quarter": f"{np.trace(tm) / tm.sum():.0%}",
        "Zero cells in the transition matrix": str(int((tm == 0).sum())),
        "Persistence: test Macro-F1": f"{pers.macro_f1:.3f}",
        "Best classifier − persistence (95% CI)": f"{best['Macro-F1 − persistence']:+.3f} [{best.ci_low:+.3f}, {best.ci_high:+.3f}] ({best.model} {best.config})",
        "Classifiers significantly better than persistence": f"{int(((vp['p Holm'] < 0.05) & (vp['Macro-F1 − persistence'] > 0)).sum())} of 6",
        "State changers predicted correctly": f"{mod.changers_accuracy.min():.0%}–{mod.changers_accuracy.max():.0%}",
        "H1 supported (Holm)": ", ".join(h1s) if h1s else "no",
        "H2 supported (Holm)": ", ".join(h2s) if h2s else "no",
        "Share of lost revenue from downgrades": f"{dec.iloc[1]:.0%}",
        "L_stop captured at top 10%: state model / stop model": f"{c10('State-model'):.1%} / {c10('Stop-model'):.1%}",
    }
comp = pd.DataFrame(cols).rename_axis("measure")
comp.to_csv(EXP / "outputs" / "tables" / "comparison_main_window_k4_k3.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60); print(comp.to_string())
