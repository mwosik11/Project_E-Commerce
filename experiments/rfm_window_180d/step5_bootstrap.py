"""Step 5 — Paired customer-level bootstrap: H1, H2 and comparison with persistence (same logic as notebook 05)."""
from common import *
import numpy as np, pandas as pd
from config import TARGET, N_STATES, RANDOM_STATE
import bootstrap as BS

N_BOOT, ALPHA, MODEL_NAMES = 5000, 0.05, ["LogReg", "RandomForest", "XGBoost"]
d = pd.read_parquet(DATA_K / "test_predictions_window180.parquet")
y, seg, cust = d[TARGET].astype(int).values, d["Segment_t"].astype(int).values, d["CustomerID"].values
uniq = np.unique(cust); W = BS.multiplicities(len(uniq), N_BOOT, RANDOM_STATE)
COLS = {("Persistence", "–"): "pred_Persistence_bench", **{(m, c): f"pred_{m}_{c}" for m in MODEL_NAMES for c in ["A", "B"]}}

def boot(mask=None):
    msk = np.ones(len(d), bool) if mask is None else mask
    pt, dr = {}, {}
    for key, col in COLS.items():
        u, Cm = BS.customer_confusions(cust[msk], y[msk], d.loc[msk, col].astype(int).values, N_STATES)
        Cf = np.zeros((len(uniq), N_STATES * N_STATES)); Cf[np.searchsorted(uniq, u)] = Cm
        pt[key] = {k: float(v) for k, v in BS.cm_metrics(Cf.sum(0), N_STATES).items()}; dr[key] = BS.cm_metrics(W @ Cf, N_STATES)
    return pt, dr
point, draws = boot()
p4 = pd.read_csv(TABLES / "step4_test_results.csv")
for (m, c), v in point.items():
    assert abs(p4[(p4.model == m) & (p4.config == c)]["macro_f1"].item() - v["macro_f1"]) < 1e-9
delta = lambda met, m, pt=point, dr=draws: (pt[(m, "B")][met] - pt[(m, "A")][met], dr[(m, "B")][met] - dr[(m, "A")][met])

h1 = pd.DataFrame([{"classifier": m, **BS.summarize(*delta("macro_f1", m), alternative="greater")} for m in MODEL_NAMES]).rename(columns={"estimate": "Δ (B − A)", "p_value": "p (one-sided)"})
h1["p Holm"] = BS.holm(h1["p (one-sided)"]); h1["H1 supported?"] = np.where(h1["p Holm"] < ALPHA, "yes", "no")
pairs = [("RandomForest", "LogReg"), ("XGBoost", "RandomForest"), ("XGBoost", "LogReg")]
dl = {m: delta("macro_f1", m) for m in MODEL_NAMES}
h2 = pd.DataFrame([{"comparison": f"Δ_{a} − Δ_{b}", **BS.summarize(dl[a][0] - dl[b][0], dl[a][1] - dl[b][1], "two-sided")} for a, b in pairs]).rename(columns={"estimate": "difference", "p_value": "p (two-sided)"})
h2["p Holm"] = BS.holm(h2["p (two-sided)"])
vp = pd.DataFrame([{"model": m, "config": c, **BS.summarize(point[(m, c)]["macro_f1"] - point[("Persistence", "–")]["macro_f1"],
                     draws[(m, c)]["macro_f1"] - draws[("Persistence", "–")]["macro_f1"], "two-sided")} for m in MODEL_NAMES for c in ["A", "B"]]).rename(columns={"estimate": "Macro-F1 − persistence", "p_value": "p (two-sided)"})
vp["p Holm"] = BS.holm(vp["p (two-sided)"])
ch = y != seg
pt_c, dr_c = boot(ch)
sec = pd.DataFrame([{"subset": "state changers", "metric": "macro_f1", "classifier": m, **BS.summarize(*delta("macro_f1", m, pt_c, dr_c), "two-sided")} for m in MODEL_NAMES])
h1.to_csv(TABLES / "step5_h1.csv", index=False); h2.to_csv(TABLES / "step5_h2.csv", index=False)
vp.to_csv(TABLES / "step5_vs_persistence.csv", index=False); sec.to_csv(TABLES / "step5_secondary_changers.csv", index=False)
for t in (h1, h2, vp, sec): print(t.round(4).to_string())
